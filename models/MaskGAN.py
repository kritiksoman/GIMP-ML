import sys

import numpy as np
import torch
import torch.hub
from PIL import Image
from torchvision import transforms

from _model_base import ModelBase, handle_alpha

colors = np.array([
    [0, 0, 0], [204, 0, 0], [76, 153, 0],
    [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
    [51, 255, 255], [102, 51, 0], [255, 0, 0], [102, 204, 0],
    [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153],
    [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]
], dtype=np.uint8)


def mask_colors_to_indices(mask):
    x = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(colors):
        x[np.all(mask == color, axis=2), :] = idx
    return x


def scale_to_width_transform(width, method):
    def f(img):
        ow, oh = img.size
        if ow == width:
            return img
        w = width
        h = int(width * oh / ow)
        return img.resize((w, h), method)

    return f


def get_img_transform(target_width):
    return transforms.Compose([
        Image.fromarray,
        scale_to_width_transform(target_width, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        lambda x: x.unsqueeze(0)
    ])


def get_mask_transform(target_width):
    return transforms.Compose([
        mask_colors_to_indices,
        Image.fromarray,
        scale_to_width_transform(target_width, Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1 / 255., 1 / 255., 1 / 255.)),
        lambda x: x.unsqueeze(0)
    ])


class MaskGAN(ModelBase):
    def __init__(self):
        super().__init__()
        self.hub_repo = 'valgur/CelebAMask-HQ'

    def load_model(self):
        device = torch.device(self.device)
        if device.type == "cpu":
            raise RuntimeError("MaskGAN does not support CPU inference")
        gpu_ids = [device.index or 0]
        model = torch.hub.load(self.hub_repo, 'Pix2PixHD',
                               pretrained=True, map_location=self.device, gpu_ids=gpu_ids)
        model.to(self.device)
        return model

    @handle_alpha
    @torch.no_grad()
    def predict(self, img, mask, mask_m):
        h, w, d = img.shape
        assert d == 3, "Input image must be RGB"
        assert img.shape == mask.shape
        assert img.shape == mask_m.shape

        opt = self.model.opt
        transform_mask = get_mask_transform(opt.loadSize)
        transform_image = get_img_transform(opt.loadSize)
        mask = transform_mask(mask).to(self.device)
        mask_m = transform_mask(mask_m).to(self.device)
        img = transform_image(img).to(self.device)

        generated = self.model.inference(mask_m, mask, img)

        result = generated.squeeze().permute(1, 2, 0)
        result = (result + 1) * 127.5
        result = result.clamp(0, 255).byte().cpu().numpy()
        result = Image.fromarray(result)
        result = result.resize([w, h])
        return np.array(result)


model = MaskGAN()

if __name__ == '__main__':
    rpc_url = sys.argv[1]
    model.process_rpc(rpc_url)
