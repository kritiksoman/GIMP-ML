import os
baseLoc = os.path.dirname(os.path.realpath(__file__)) + '/'
from gimpfu import *
import sys
activate_this = os.path.join(baseLoc, 'gimpenv', 'bin', 'activate_this.py')
with open(activate_this) as f:
    code = compile(f.read(), activate_this, 'exec')
    exec(code, dict(__file__=activate_this))

import cv2
import numpy as np

def channelData(layer):  # convert gimp image to numpy
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    bpp = region.bpp
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(layer.height, layer.width, bpp)

def createResultLayer(image, name, result):
    rlBytes = np.uint8(result).tobytes();
    rl = gimp.Layer(image, name, image.width, image.height, image.active_layer.type, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()

def genNewImg(name, layer_np):
    h, w, d = layer_np.shape
    img = pdb.gimp_image_new(w, h, RGB)
    display = pdb.gimp_display_new(img)

    rlBytes = np.uint8(layer_np).tobytes();
    rl = gimp.Layer(img, name, img.width, img.height, RGB, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes

    pdb.gimp_image_insert_layer(img, rl, None, 0)

    gimp.displays_flush()


def colorpalette():
    cpy = cv2.cvtColor(cv2.imread(baseLoc+'color_palette.png'),cv2.COLOR_BGR2RGB)
    genNewImg('palette', cpy)


register(
    "colorpalette",
    "colorpalette",
    "colorpalette.",
    "Kritik Soman",
    "Your",
    "2020",
    "colorpalette...",
    "",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [],
    [],
    colorpalette, menu="<Image>/Layer/GIML-ML")

main()
