from _util import add_gimpenv_to_pythonpath, tqdm_as_gimp_progress

add_gimpenv_to_pythonpath()

from gimpfu import *
import numpy as np
import torch
import torch.hub
from PIL import Image
from torchvision import transforms

colors = np.array([
    [0, 0, 0],
    [204, 0, 0],
    [0, 255, 255],
    [51, 255, 255],
    [51, 51, 255],
    [204, 0, 204],
    [204, 204, 0],
    [102, 51, 0],
    [255, 0, 0],
    [0, 204, 204],
    [76, 153, 0],
    [102, 204, 0],
    [255, 255, 0],
    [0, 0, 153],
    [255, 153, 51],
    [0, 51, 0],
    [0, 204, 0],
    [0, 0, 204],
    [255, 51, 153]
], dtype=np.uint8)


@tqdm_as_gimp_progress("Downloading model")
def load_model(device):
    net = torch.hub.load('valgur/face-parsing.PyTorch', 'BiSeNet', pretrained=True, map_location=device)
    net.to(device)
    return net


@torch.no_grad()
def getface(input_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(device)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    input_image = Image.fromarray(input_image)
    img = input_image.resize((512, 512), Image.BICUBIC)
    img = to_tensor(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)

    out = net(img)[0]

    result = out.squeeze(0).argmax(0).byte().cpu().numpy()
    result = Image.fromarray(result).resize(input_image.size)
    result.putpalette(colors.tobytes())
    result = np.array(result.convert('RGB'))
    return result


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


def faceparse(img, layer):
    if torch.cuda.is_available():
        gimp.progress_init("(Using GPU) Running face parse for " + layer.name + "...")
    else:
        gimp.progress_init("(Using CPU) Running face parse for " + layer.name + "...")

    imgmat = channelData(layer)
    cpy = getface(imgmat)
    createResultLayer(img, 'new_output', cpy)


register(
    "faceparse",
    "faceparse",
    "Running face parse.",
    "Kritik Soman",
    "Your",
    "2020",
    "faceparse...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    faceparse, menu="<Image>/Layer/GIML-ML")

main()
