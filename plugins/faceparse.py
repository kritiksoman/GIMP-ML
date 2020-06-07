import sys
from os.path import dirname, realpath

sys.path.append(realpath(dirname(__file__)))
from gimpfu import main
from _plugin_base import GimpPluginBase


class FaceParse(GimpPluginBase):
    def run(self):
        self.model_file = 'FaceParse_BiSeNet.py'
        result = self.predict(self.drawable)
        self.create_layer(result)


plugin = FaceParse()
plugin.register(
    proc_name="faceparse",
    blurb="faceparse",
    help="Generate semantic segmentation for facial features.",
    author="Kritik Soman",
    copyright="",
    date="2020",
    label="faceparse...",
    imagetypes="RGB*"
)
main()
