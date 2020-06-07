import sys
from os.path import dirname, realpath

sys.path.append(realpath(dirname(__file__)))
from gimpfu import main
from _plugin_base import GimpPluginBase


class Colorize(GimpPluginBase):
    def run(self):
        self.model_file = 'NeuralColorization.py'
        result = self.predict(self.drawable)
        self.create_image(result)


plugin = Colorize()
plugin.register(
    proc_name="colorize",
    blurb="colorize",
    help="Colorize grayscale images",
    author="Kritik Soman",
    copyright="",
    date="2020",
    label="colorize...",
    imagetypes="GRAY*"
)
main()
