import sys
from os.path import dirname, realpath

sys.path.append(realpath(dirname(__file__)))
from gimpfu import main
from _plugin_base import GimpPluginBase


class MonoDepth(GimpPluginBase):
    def run(self):
        self.model_file = 'Monodepth2.py'
        result = self.predict(self.drawable)
        self.create_layer(result)


plugin = MonoDepth()
plugin.register(
    proc_name="MonoDepth",
    blurb="MonoDepth",
    help="Generate monocular disparity map based on deep learning.",
    author="Kritik Soman",
    copyright="",
    date="2020",
    label="MonoDepth...",
    imagetypes="RGB*"
)
main()
