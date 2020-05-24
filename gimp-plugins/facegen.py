from _util import add_gimpenv_to_pythonpath, tqdm_as_gimp_progress

add_gimpenv_to_pythonpath()

from gimpfu import *
import numpy as np
import torch
import torch.hub
import torchvision.transforms as transforms
from PIL import Image

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


@tqdm_as_gimp_progress("Downloading model")
def load_model(device):
    device = torch.device(device)
    if device.type == "cpu":
        raise RuntimeError("MaskGAN does not support CPU inference")
    gpu_ids = [device.index or 0]
    model = torch.hub.load('valgur/CelebAMask-HQ', 'Pix2PixHD',
                           pretrained=True, map_location=device, gpu_ids=gpu_ids)
    model.to(device)
    return model


@torch.no_grad()
def getnewface(img, mask, mask_m):
    h, w, d = img.shape
    assert d == 3, "Input image must be RGB"
    assert img.shape == mask.shape
    assert img.shape == mask_m.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    opt = model.opt
    transform_mask = get_mask_transform(opt.loadSize)
    transform_image = get_img_transform(opt.loadSize)
    mask = transform_mask(mask).to(device)
    mask_m = transform_mask(mask_m).to(device)
    img = transform_image(img).to(device)

    generated = model.inference(mask_m, mask, img)

    result = generated.squeeze().permute(1, 2, 0)
    result = (result + 1) * 127.5
    result = result.clamp(0, 255).byte().cpu().numpy()
    result = Image.fromarray(result)
    result = result.resize([w, h])
    return np.array(result)


def channelData(layer):  # convert gimp image to numpy
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    bpp = region.bpp
    # return np.frombuffer(pixChars,dtype=np.uint8).reshape(len(pixChars)/bpp,bpp)
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(layer.height, layer.width, bpp)


def createResultLayer(image, name, result):
    rlBytes = np.uint8(result).tobytes()
    rl = gimp.Layer(image, name, image.width, image.height, image.active_layer.type, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()


def facegen(imggimp, curlayer, layeri, layerm, layermm):
    if torch.cuda.is_available():
        gimp.progress_init("(Using GPU) Running face gen for " + layeri.name + "...")
    else:
        gimp.progress_init("(Using CPU) Running face gen for " + layeri.name + "...")

    img = channelData(layeri)
    mask = channelData(layerm)
    mask_m = channelData(layermm)

    cpy = getnewface(img, mask, mask_m)
    createResultLayer(imggimp, 'new_output', cpy)


register(
    "facegen",
    "facegen",
    "Running face gen...",
    "Kritik Soman",
    "Your",
    "2020",
    "facegen...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
        (PF_LAYER, "drawinglayer", "Original Image:", None),
        (PF_LAYER, "drawinglayer", "Original Mask:", None),
        (PF_LAYER, "drawinglayer", "Modified Mask:", None),
    ],
    [],
    facegen, menu="<Image>/Layer/GIML-ML")

main()
