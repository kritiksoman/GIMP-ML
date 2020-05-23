from _util import *

add_gimpenv_to_pythonpath()

from gimpfu import register, main, gimp
import gimpfu as gfu
import torch
import torch.hub
import numpy as np
from PIL import Image


@tqdm_as_gimp_progress("Downloading model")
def load_model(device):
    model = torch.hub.load('valgur/pytorch-SRResNet', 'SRResNet', pretrained=True, map_location=device)
    model.to(device)
    return model


@torch.no_grad()
def super_resolution(input_image, device="cuda"):
    h, w, d = input_image.shape
    assert d == 3, "Input image must be RGB"

    model = load_model(device)

    im_input = torch.from_numpy(input_image).permute(2, 0, 1).to(device)
    im_input = im_input.float().div(255).unsqueeze(0)

    HR_4x = model(im_input).squeeze().permute(1, 2, 0)
    return HR_4x.clamp(0, 1).mul(255).byte().cpu().numpy()


def process(img, layer):
    gimp.progress_init("(Using {}) Upscaling {}...".format(
        "GPU" if default_device().type == "cuda" else "CPU",
        layer.name
    ))

    rgb, alpha = split_alpha(layer_to_numpy(layer))
    result = super_resolution(rgb, default_device())
    if alpha is not None:
        h, w, d = result.shape
        alpha = np.array(Image.fromarray(alpha[..., 0]).resize((w, h), Image.BILINEAR))[..., None]
    result = merge_alpha(result, alpha)
    numpy_to_gimp_image(result, layer.name + '_upscaled')


register(
    "super-resolution",
    "super-resolution",
    "Running super-resolution.",
    "Kritik Soman",
    "",
    "2020",
    "super-resolution...",
    "RGB*",
    [
        (gfu.PF_IMAGE, "image", "Input image", None),
        (gfu.PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    process,
    menu="<Image>/Layer/GIML-ML")

main()
