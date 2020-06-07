import sys
from os.path import dirname, realpath

sys.path.append(realpath(dirname(__file__)))
import gimpfu as gfu
from gimpfu import main
from _plugin_base import GimpPluginBase


class FaceGen(GimpPluginBase):
    def run(self, img_layer, mask_layer, mask_m_layer):
        self.model_file = 'MaskGAN.py'
        result = self.predict(img_layer, mask_layer, mask_m_layer)
        self.create_layer(result)


plugin = FaceGen()
plugin.register(
    proc_name="facegen",
    blurb="facegen",
    help="Running face gen...",
    author="Kritik Soman",
    copyright="",
    date="2020",
    label="facegen...",
    imagetypes="RGB*",
    params=
    [
        (gfu.PF_LAYER, "drawinglayer", "Original Image:", None),
        (gfu.PF_LAYER, "drawinglayer", "Original Mask:", None),
        (gfu.PF_LAYER, "drawinglayer", "Modified Mask:", None),
    ]
)
main()
