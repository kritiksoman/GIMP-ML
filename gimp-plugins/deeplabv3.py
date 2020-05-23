from _util import add_gimpenv_to_pythonpath, tqdm_as_gimp_progress

add_gimpenv_to_pythonpath()

from gimpfu import *
import numpy as np
import torch
import torch.hub
from PIL import Image
from torchvision import transforms


@tqdm_as_gimp_progress("Downloading model")
def load_model(device):
    model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def getSeg(input_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    h, w, _ = input_image.shape
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)

    output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)

    # apply a color palette, selecting a color for each class
    palette = np.array([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = np.arange(21)[:, None] * palette
    colors = (colors % 255).astype(np.uint8)
    result = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((w, h))
    result.putpalette(colors.tobytes())
    return np.array(result.convert('RGB'))


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


def deeplabv3(img, layer):
    if torch.cuda.is_available():
        gimp.progress_init("(Using GPU) Generating semantic segmentation map for " + layer.name + "...")
    else:
        gimp.progress_init("(Using CPU) Generating semantic segmentation map for " + layer.name + "...")

    imgmat = channelData(layer)
    cpy = getSeg(imgmat)
    createResultLayer(img, 'new_output', cpy)


register(
    "deeplabv3",
    "deeplabv3",
    "Generate semantic segmentation map based on deep learning.",
    "Kritik Soman",
    "GIMP-ML",
    "2020",
    "deeplabv3...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None)
    ],
    [],
    deeplabv3, menu="<Image>/Layer/GIML-ML")

main()
