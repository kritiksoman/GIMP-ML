from _util import add_gimpenv_to_pythonpath, tqdm_as_gimp_progress

add_gimpenv_to_pythonpath()

from gimpfu import *
import numpy as np
import torch.hub


@tqdm_as_gimp_progress("Downloading model")
def load_model(device):
    predictor = torch.hub.load('valgur/DeblurGANv2:python2', 'predictor', 'fpn_inception', device=device)
    return predictor


def getdeblur(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = load_model(device)
    pred = predictor(img, None)
    return pred


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


def deblur(img, layer):
    gimp.progress_init("Running for " + layer.name + "...")
    imgmat = channelData(layer)
    pred = getdeblur(imgmat)
    createResultLayer(img, 'deblur_' + layer.name, pred)


register(
    "deblur",
    "deblur",
    "Running deblurring.",
    "Kritik Soman",
    "Your",
    "2020",
    "deblur...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    deblur, menu="<Image>/Layer/GIML-ML")

main()
