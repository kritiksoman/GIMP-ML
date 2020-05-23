from _util import add_gimpenv_to_pythonpath, tqdm_as_gimp_progress

add_gimpenv_to_pythonpath()

from gimpfu import *
import torch
import numpy as np
import torch.hub


@tqdm_as_gimp_progress("Downloading model")
def load_model(device):
    model = torch.hub.load('valgur/pytorch-SRResNet', 'SRResNet', pretrained=True, map_location=device)
    model.to(device)
    return model


@torch.no_grad()
def getnewimg(input_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    im_input = torch.from_numpy(input_image).permute(2, 0, 1).to(device)
    im_input = im_input.float().div(255).unsqueeze(0)

    HR_4x = model(im_input).squeeze().permute(1, 2, 0)
    return HR_4x.clamp(0, 1).mul(255).byte().cpu().numpy()


def channelData(layer):  # convert gimp image to numpy
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    bpp = region.bpp
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(layer.height, layer.width, bpp)


def createResultLayer(image, name, result):
    rlBytes = np.uint8(result).tobytes()
    rl = gimp.Layer(image, name, image.width, image.height, image.active_layer.type, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()


def genNewImg(name, layer_np):
    h, w, d = layer_np.shape
    img = pdb.gimp_image_new(w, h, RGB)
    display = pdb.gimp_display_new(img)

    rlBytes = np.uint8(layer_np).tobytes()
    rl = gimp.Layer(img, name, img.width, img.height, RGB, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes

    pdb.gimp_image_insert_layer(img, rl, None, 0)

    gimp.displays_flush()


def super_resolution(img, layer):
    if torch.cuda.is_available():
        gimp.progress_init("(Using GPU) Running for " + layer.name + "...")
    else:
        gimp.progress_init("(Using CPU) Running for " + layer.name + "...")

    imgmat = channelData(layer)
    cpy = getnewimg(imgmat)
    genNewImg(layer.name + '_upscaled', cpy)


register(
    "super-resolution",
    "super-resolution",
    "Running super-resolution.",
    "Kritik Soman",
    "Your",
    "2020",
    "super-resolution...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    super_resolution, menu="<Image>/Layer/GIML-ML")

main()
