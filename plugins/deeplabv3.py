from gimpfu import main

from _plugin_base import GimpPluginBase


class DeepLabV3(GimpPluginBase):
    def run(self):
        self.model_file = 'DeepLabV3.py'
        result = self.predict(self.drawable)
        self.create_layer(result)


plugin = DeepLabV3()
plugin.register(
    proc_name="deeplabv3",
    blurb="deeplabv3",
    help="Generate semantic segmentation map based on deep learning.",
    author="Kritik Soman",
    copyright="",
    date="2020",
    label="deeplabv3...",
    imagetypes="RGB*"
)
main()
