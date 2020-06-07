from __future__ import print_function, absolute_import, division

import os
import subprocess
import sys
import threading
from SimpleXMLRPCServer import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
from abc import ABCMeta, abstractmethod
from xmlrpclib import Binary

import gimpfu as gfu
from gimpfu import gimp, pdb

from _config import python3_executable

base_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
models_dir = os.path.join(base_dir, 'models')


class GimpPluginBase(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.model_file = None
        self.gimp_img = None
        self.drawable = None
        self.name = None

    @abstractmethod
    def run(self, *args, **kwargs):
        # example
        self.model_file = 'xyz.py'
        result = self.predict(self.drawable)
        self.create_layer(result)

    def create_layer(self, result, name=None):
        name = name or self.drawable.name + ' ' + self.name
        imgarray_to_layer(result, self.gimp_img, name)

    def create_image(self, result, name=None):
        name = name or self.drawable.name + ' ' + self.name
        imgarray_to_image(result, name)

    def register(self, proc_name, blurb, help, author, copyright, date, label,
                 imagetypes, params=None, results=None, menu="<Image>/Layer/GIML-ML",
                 domain=None, on_query=None, on_run=None):
        self.name = proc_name
        gfu.register(
            proc_name,
            blurb,
            help,
            author,
            copyright,
            date,
            label,
            imagetypes,
            params=[(gfu.PF_IMAGE, "image", "Input image", None),
                    (gfu.PF_DRAWABLE, "drawable", "Input drawable", None)]
                   + (params or []),
            results=results or [],
            function=self.run_outer,
            menu=menu,
            domain=domain, on_query=on_query, on_run=on_run
        )

    def run_outer(self, gimp_img, drawable, *extra_args):
        self.gimp_img = gimp_img
        self.drawable = drawable
        print("Running {}...".format(self.name))
        gimp.progress_init("Running {}...".format(self.name))
        self.run(*extra_args)

    def predict(self, *args, **kwargs):
        assert self.model_file is not None
        model_proxy = ModelProxy(self.model_file)
        return model_proxy(*args, **kwargs)


class ModelProxy(object):
    """
    When called, runs
        python3 models/<model_file>
    and waits for the subprocess to call get_args() and then return_result() over XML-RPC.
    Additionally, any progress info can be sent via update_progress().
    """

    def __init__(self, model_file):
        self.python_executable = python3_executable
        self.model_path = os.path.join(models_dir, model_file)
        self.server = None
        self.args = None
        self.kwargs = None
        self.result = None

    @staticmethod
    def _encode(x):
        if isinstance(x, gimp.Layer):
            x = layer_to_imgarray(x)
        if isinstance(x, ImgArray):
            x = x.encode()
        return x

    @staticmethod
    def _decode(x):
        if isinstance(x, list) and len(x) == 2 and hasattr(x[0], 'data'):
            x = ImgArray.decode(x)
        return x

    def _rpc_get_args(self):
        assert isinstance(self.args, (list, tuple))
        assert isinstance(self.kwargs, dict)
        args = [self._encode(arg) for arg in self.args]
        kwargs = {k: self._encode(v) for k, v in self.kwargs.items()}
        return args, kwargs

    def _rpc_return_result(self, result):
        assert isinstance(result, (list, tuple))
        self.result = tuple(self._decode(x) for x in result)

    def _subproc_thread(self, rpc_port):
        env = self._add_conda_env_to_path()
        try:
            self.proc = subprocess.Popen([
                self.python_executable,
                self.model_path,
                'http://127.0.0.1:{}/'.format(rpc_port)
            ], env=env)
            self.proc.wait()
        finally:
            self.server.shutdown()
            self.server.server_close()

    def _add_conda_env_to_path(self):
        env = os.environ.copy()
        conda_root = os.path.dirname(self.python_executable)
        env['PATH'] = os.pathsep.join([
            conda_root,
            os.path.join(conda_root, 'Library', 'mingw-w64', 'bin'),
            os.path.join(conda_root, 'Library', 'usr', 'bin'),
            os.path.join(conda_root, 'Library', 'bin'),
            os.path.join(conda_root, 'Scripts'),
            os.path.join(conda_root, 'bin'),
            env['PATH']
        ])
        return env

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        # For cleaner exception info
        class RequestHandler(SimpleXMLRPCRequestHandler):
            def _dispatch(self, method, params):
                try:
                    return self.server.funcs[method](*params)
                except:
                    self.server.exception = sys.exc_info()
                    raise

        self.server = SimpleXMLRPCServer(('127.0.0.1', 0), allow_none=True, logRequests=False,
                                         requestHandler=RequestHandler)
        rpc_port = self.server.server_address[1]
        self.server.register_function(self._rpc_get_args, 'get_args')
        self.server.register_function(self._rpc_return_result, 'return_result')
        self.server.register_function(update_progress)

        t = threading.Thread(target=self._subproc_thread, args=(rpc_port,))
        t.start()
        self.server.exception = None
        self.server.serve_forever()

        if self.result is None:
            if self.server.exception:
                type, value, traceback = self.server.exception
                raise type, value, traceback
            raise RuntimeError("Model did not return a result!")
        if len(self.result) == 1:
            return self.result[0]
        return self.result


class ImgArray(object):
    """Minimal Numpy ndarray-like object for serialization in RPC."""

    def __init__(self, buffer, shape):
        self.buffer = buffer
        self.shape = shape

    def encode(self):
        return Binary(self.buffer), self.shape

    @staticmethod
    def decode(x):
        return ImgArray(x[0].data, x[1])


image_type_map = {
    1: gfu.GRAY_IMAGE,
    2: gfu.GRAYA_IMAGE,
    3: gfu.RGB_IMAGE,
    4: gfu.RGBA_IMAGE,
}

image_base_type_map = {
    1: gfu.GRAY,
    2: gfu.GRAY,
    3: gfu.RGB,
    4: gfu.RGB,
}


def layer_to_imgarray(layer):
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    return ImgArray(pixChars, (layer.height, layer.width, region.bpp))


def imgarray_to_layer(array, gimp_img, name):
    h, w, d = array.shape
    layer = gimp.Layer(gimp_img, name, w, h, image_type_map[d])
    region = layer.get_pixel_rgn(0, 0, w, h)
    region[:, :] = array.buffer
    gimp_img.insert_layer(layer, position=0)
    return layer


def imgarray_to_image(array, name):
    h, w, d = array.shape
    img = gimp.Image(w, h, image_base_type_map[d])
    imgarray_to_layer(array, img, name)
    gimp.Display(img)
    gimp.displays_flush()


def update_progress(percent, message):
    if percent is not None:
        pdb.gimp_progress_update(percent)
    else:
        pdb.gimp_progress_pulse()
    pdb.gimp_progress_set_text(message)
