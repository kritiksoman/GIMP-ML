import sys

import numpy as np
import torch
from PIL import Image

from _model_base import ModelBase, handle_alpha


def yuv2rgb(yuv):
    yuv_from_rgb = np.array([
        [0.299, 0.587, 0.114],
        [-0.14714119, -0.28886916, 0.43601035],
        [0.61497538, -0.51496512, -0.10001026]
    ])
    rgb_from_yuv = np.linalg.inv(yuv_from_rgb)
    return np.dot(yuv, rgb_from_yuv.T.copy())


class NeuralColorization(ModelBase):
    def __init__(self):
        super().__init__()
        self.hub_repo = 'valgur/neural-colorization:pytorch'

    def load_model(self):
        G = torch.hub.load(self.hub_repo, 'generator',
                           pretrained=True, map_location=self.device)
        G.to(self.device)
        return G

    @handle_alpha
    @torch.no_grad()
    def predict(self, img):
        h, w, d = img.shape
        assert d == 1, "Input image must be grayscale"
        G = self.model

        input_image = img[..., 0]
        H, W = input_image.shape
        infimg = input_image[None, None, ...].astype(np.float32) / 255.
        img_variable = torch.from_numpy(infimg) - 0.5
        img_variable = img_variable.to(self.device)
        res = G(img_variable)
        uv = res.cpu().numpy()[0]
        uv[0, :, :] *= 0.436
        uv[1, :, :] *= 0.615
        u = np.array(Image.fromarray(uv[0]).resize((W, H), Image.BILINEAR))[None, ...]
        v = np.array(Image.fromarray(uv[1]).resize((W, H), Image.BILINEAR))[None, ...]
        yuv = np.concatenate([infimg[0], u, v], axis=0).transpose(1, 2, 0)

        rgb = yuv2rgb(yuv * 255)
        rgb = rgb.clip(0, 255).astype(np.uint8)
        return rgb


model = NeuralColorization()

if __name__ == '__main__':
    rpc_url = sys.argv[1]
    model.process_rpc(rpc_url)
