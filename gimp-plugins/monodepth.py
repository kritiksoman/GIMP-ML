import os

baseLoc = os.path.dirname(os.path.realpath(__file__)) + '/'

from gimpfu import *
import sys

sys.path.extend([baseLoc + 'gimpenv/lib/python2.7', baseLoc + 'gimpenv/lib/python2.7/site-packages',
                 baseLoc + 'gimpenv/lib/python2.7/site-packages/setuptools', baseLoc + 'MiDaS'])

from run import run_depth
from monodepth_net import MonoDepthNet
import MiDaS_utils as MiDaS_utils
import numpy as np
import cv2
import torch

def getMonoDepth(input_image,cFlag):
    image = input_image / 255.0
    out = run_depth(image, baseLoc+'weights/MiDaS/model.pt', MonoDepthNet, MiDaS_utils, target_w=640,f=cFlag)
    out = np.repeat(out[:, :, np.newaxis], 3, axis=2)
    d1,d2 = input_image.shape[:2]
    out = cv2.resize(out,(d2,d1))
    # cv2.imwrite("/Users/kritiksoman/PycharmProjects/new/out.png", out)
    return out


def channelData(layer):  # convert gimp image to numpy
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    bpp = region.bpp
    # return np.frombuffer(pixChars,dtype=np.uint8).reshape(len(pixChars)/bpp,bpp)
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(layer.height, layer.width, bpp)


def createResultLayer(image, name, result):
    rlBytes = np.uint8(result).tobytes();
    rl = gimp.Layer(image, name, image.width, image.height, 0, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()


def MonoDepth(img, layer,cFlag):
    imgmat = channelData(layer)
    if imgmat.shape[0] != img.height or imgmat.shape[1] != img.width:
        pdb.gimp_message(" Do (Layer -> Layer to Image Size) first and try again.")
    else:
        if torch.cuda.is_available() and not cFlag:
            gimp.progress_init("(Using GPU) Generating disparity map for " + layer.name + "...")
        else:
            gimp.progress_init("(Using CPU) Generating disparity map for " + layer.name + "...")

        if imgmat.shape[2] == 4:  # get rid of alpha channel
            imgmat = imgmat[:,:,0:3]
        cpy = getMonoDepth(imgmat,cFlag)
        createResultLayer(img, 'new_output', cpy)


register(
    "MonoDepth",
    "MonoDepth",
    "Generate monocular disparity map based on deep learning.",
    "Kritik Soman",
    "Your",
    "2020",
    "MonoDepth...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [(PF_IMAGE, "image", "Input image", None),
     (PF_DRAWABLE, "drawable", "Input drawable", None),
     (PF_BOOL, "fcpu", "Force CPU", False)
     ],
    [],
    MonoDepth, menu="<Image>/Layer/GIML-ML")

main()