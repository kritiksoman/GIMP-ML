import sys
from os.path import dirname, realpath

sys.path.append(realpath(dirname(__file__)))
from gimpfu import main
from _plugin_base import GimpPluginBase


class SuperResolution(GimpPluginBase):
    def run(self):
        self.model_file = 'SRResNet.py'
        result = self.predict(self.drawable)
        self.create_image(result)


plugin = SuperResolution()
plugin.register(
    proc_name="super-resolution",
    blurb="super-resolution",
    help="Running super-resolution.",
    author="Kritik Soman",
    copyright="",
    date="2020",
    label="super-resolution...",
    imagetypes="RGB*"
)
main()
