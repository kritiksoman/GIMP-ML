import sys

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import torch.hub
from PIL import Image
from torchvision import transforms

from _model_base import ModelBase, handle_alpha


class Monodepth2(ModelBase):
    def __init__(self):
        super().__init__()
        self.hub_repo = "valgur/monodepth2"

    def load_model(self):
        pretrained_model = "mono+stereo_640x192"
        encoder = torch.hub.load(self.hub_repo, "ResnetEncoder", pretrained_model, map_location=self.device)
        depth_decoder = torch.hub.load(self.hub_repo, "DepthDecoder", pretrained_model, map_location=self.device)
        encoder.to(self.device)
        depth_decoder.to(self.device)
        return depth_decoder, encoder

    @handle_alpha
    @torch.no_grad()
    def predict(self, input_image):
        h, w, d = input_image.shape
        assert d == 3, "Input image must be RGB"

        # LOADING PRETRAINED MODEL
        depth_decoder, encoder = self.model

        input_image = Image.fromarray(input_image)
        original_width, original_height = input_image.size
        input_image = input_image.resize((encoder.feed_width, encoder.feed_height), Image.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.to(self.device)

        # PREDICTION
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Convert to colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        return colormapped_im


model = Monodepth2()

if __name__ == '__main__':
    rpc_url = sys.argv[1]
    model.process_rpc(rpc_url)
