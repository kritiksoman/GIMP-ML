import os

baseLoc = os.path.dirname(os.path.realpath(__file__)) + '/'

from gimpfu import *
import sys

sys.path.extend([baseLoc + 'gimpenv/lib/python2.7', baseLoc + 'gimpenv/lib/python2.7/site-packages',
                 baseLoc + 'gimpenv/lib/python2.7/site-packages/setuptools', baseLoc + 'PyTorch-Image-Dehazing'])


import torch
import net
import numpy as np
import cv2

def clrImg(data_hazy,cFlag):
    data_hazy = (data_hazy / 255.0)
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    dehaze_net = net.dehaze_net()

    if torch.cuda.is_available() and not cFlag:
        dehaze_net = dehaze_net.cuda()
        dehaze_net.load_state_dict(torch.load(baseLoc+'weights/deepdehaze/dehazer.pth'))
        data_hazy = data_hazy.cuda()
    else:
        dehaze_net.load_state_dict(torch.load(baseLoc+'weights/deepdehaze/dehazer.pth',map_location=torch.device("cpu")))

    gimp.progress_update(float(0.005))
    gimp.displays_flush()    
    data_hazy = data_hazy.unsqueeze(0)
    clean_image = dehaze_net(data_hazy)
    out = clean_image.detach().cpu().numpy()[0,:,:,:]*255
    out = np.clip(np.transpose(out,(1,2,0)),0,255).astype(np.uint8)
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


def deepdehazing(img, layer, cFlag):
    if torch.cuda.is_available() and not cFlag:
        gimp.progress_init("(Using GPU) Dehazing " + layer.name + "...")
    else:
        gimp.progress_init("(Using CPU) Dehazing " + layer.name + "...")
    imgmat = channelData(layer)
    if imgmat.shape[2] == 4:  # get rid of alpha channel
        imgmat = imgmat[:,:,0:3]
    cpy = clrImg(imgmat,cFlag)
    createResultLayer(img, 'new_output', cpy)


register(
    "deep-dehazing",
    "deep-dehazing",
    "Dehaze image based on deep learning.",
    "Kritik Soman",
    "Your",
    "2020",
    "deep-dehazing...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [(PF_IMAGE, "image", "Input image", None),
     (PF_DRAWABLE, "drawable", "Input drawable", None),
     (PF_BOOL, "fcpu", "Force CPU", False)
     ],
    [],
    deepdehazing, menu="<Image>/Layer/GIML-ML")

main()