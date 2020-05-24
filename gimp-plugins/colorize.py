from _util import *

add_gimpenv_to_pythonpath()

from gimpfu import register, main, gimp
import gimpfu as gfu
import numpy as np
import torch
import torch.hub
from PIL import Image


def yuv2rgb(yuv):
    yuv_from_rgb = np.array([
        [0.299, 0.587, 0.114],
        [-0.14714119, -0.28886916, 0.43601035],
        [0.61497538, -0.51496512, -0.10001026]
    ])
    rgb_from_yuv = np.linalg.inv(yuv_from_rgb)
    return np.dot(yuv, rgb_from_yuv.T.copy())


@tqdm_as_gimp_progress("Downloading model")
def load_model(device):
    G = torch.hub.load('valgur/neural-colorization:pytorch', 'generator',
                       pretrained=True, map_location=device)
    G.to(device)
    return G


@handle_alpha
@torch.no_grad()
def colorize(input_image, device="cuda"):
    h, w, d = input_image.shape
    assert d == 1, "Input image must be grayscale"
    G = load_model(device)

    input_image = input_image[..., 0]
    H, W = input_image.shape
    infimg = input_image[None, None, ...].astype(np.float32) / 255.
    img_variable = torch.from_numpy(infimg) - 0.5
    img_variable = img_variable.to(device)
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


def process(gimp_img, layer):
    gimp.progress_init("(Using {}) Colorizing {}...".format(
        "GPU" if default_device().type == "cuda" else "CPU",
        layer.name
    ))
    img = layer_to_numpy(layer)
    result = colorize(img, default_device())
    numpy_to_gimp_image(result, layer.name + '_colorized')


register(
    "colorize",
    "colorize",
    "Generate monocular disparity map based on deep learning.",
    "Kritik Soman",
    "",
    "2020",
    "colorize...",
    "GRAY*",
    [
        (gfu.PF_IMAGE, "image", "Input image", None),
        (gfu.PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    process,
    menu="<Image>/Layer/GIML-ML"
)

main()
