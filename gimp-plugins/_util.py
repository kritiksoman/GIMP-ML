import os
import sys

import gimpfu as gfu
from gimpfu import pdb, gimp

baseLoc = os.path.dirname(os.path.realpath(__file__))


def add_gimpenv_to_pythonpath():
    env_path = os.path.join(baseLoc, 'gimpenv/lib/python2.7')
    if env_path in sys.path:
        return
    # Prepend to PYTHONPATH to make sure the gimpenv packages get loaded before system ones,
    # since they are likely more up-to-date.
    sys.path[:0] = [
        env_path,
        os.path.join(env_path, 'site-packages'),
        os.path.join(env_path, 'site-packages/setuptools')
    ]


def default_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class tqdm_as_gimp_progress:
    def __init__(self, default_desc=None):
        self.default_desc = default_desc

    def __enter__(self):
        from tqdm import tqdm
        self.tqdm_display = tqdm.display

        def custom_tqdm_display(tqdm_self, *args, **kwargs):
            tqdm_info = tqdm_self.format_dict.copy()
            tqdm_info["prefix"] = tqdm_info["prefix"] or self.default_desc
            tqdm_info["bar_format"] = "{desc}: " if tqdm_info["prefix"] else ""
            # Removed {percentage:3.0f}% from bar_format
            tqdm_info["bar_format"] += "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            pdb.gimp_progress_set_text(tqdm_self.format_meter(**tqdm_info))
            if tqdm_info["total"]:
                pdb.gimp_progress_update(tqdm_info["n"] / float(tqdm_info["total"]))
            else:
                pdb.gimp_progress_pulse()
            self.tqdm_display(tqdm_self, *args, **kwargs)

        tqdm.display = custom_tqdm_display
        return self

    def __exit__(self, exit_type, exit_value, exit_traceback):
        from tqdm import tqdm
        tqdm.display = self.tqdm_display

    def __call__(self, func):
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator


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


def layer_to_numpy(layer):
    import numpy as np
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    bpp = region.bpp
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(layer.height, layer.width, bpp)


def split_alpha(array):
    h, w, d = array.shape
    if d == 1:
        return array, None
    if d == 2:
        return array[:, :, 0:1], array[:, :, 1:2]
    if d == 3:
        return array, None
    if d == 4:
        return array[:, :, 0:3], array[:, :, 3:4]
    raise ValueError("Image has too many channels ({})".format(d))


def merge_alpha(image, alpha):
    import numpy as np
    h, w, d = image.shape
    if d not in (1, 3):
        raise ValueError("Incorrect number of channels ({})".format(d))
    if alpha is None:
        return image
    return np.concatenate([image, alpha], axis=2)


def combine_alphas(alphas):
    combined_alpha = None
    for alpha in alphas:
        if alpha is not None:
            if combined_alpha is None:
                combined_alpha = alpha
            else:
                combined_alpha = combined_alpha * (alpha / 255.)
    if combined_alpha is not None:
        combined_alpha = combined_alpha.astype("uint8")
    return combined_alpha


def handle_alpha(func):
    def decorator(*args, **kwargs):
        import numpy as np

        alphas = []
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                img, alpha = split_alpha(arg)
                args[i] = img
                alphas.append(alpha)
        for key, arg in list(kwargs.items()):
            if isinstance(arg, np.ndarray):
                img, alpha = split_alpha(arg)
                kwargs[key] = img
                alphas.append(alpha)

        result = func(*args, **kwargs)
        alpha = combine_alphas(alphas)

        # for super-res
        if alpha is not None and result.shape[:2] != alpha.shape[:2]:
            from PIL import Image
            h, w, d = result.shape
            alpha = np.array(Image.fromarray(alpha[..., 0]).resize((w, h), Image.BILINEAR))[..., None]

        result = merge_alpha(result, alpha)
        return result

    return decorator


def numpy_to_layer(array, gimp_image, name):
    import numpy as np
    h, w, d = array.shape
    layer = gimp.Layer(gimp_image, name, w, h, image_type_map[d])
    region = layer.get_pixel_rgn(0, 0, w, h)
    data = array.astype(np.uint8).tobytes()
    region[:, :] = data
    gimp_image.insert_layer(layer, position=0)
    return layer


def numpy_to_gimp_image(array, name):
    h, w, d = array.shape
    img = gimp.Image(w, h, image_base_type_map[d])
    numpy_to_layer(array, img, name)
    gimp.Display(img)
    gimp.displays_flush()
