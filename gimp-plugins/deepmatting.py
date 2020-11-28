import os

baseLoc = os.path.dirname(os.path.realpath(__file__)) + '/'

from gimpfu import *
import sys

sys.path.extend([baseLoc + 'gimpenv/lib/python2.7', baseLoc + 'gimpenv/lib/python2.7/site-packages',
                 baseLoc + 'gimpenv/lib/python2.7/site-packages/setuptools', baseLoc + 'pytorch-deep-image-matting'])

import torch
from argparse import Namespace
import net
import cv2
import os
import numpy as np
from deploy import inference_img_whole


def channelData(layer):  # convert gimp image to numpy
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    bpp = region.bpp
    # return np.frombuffer(pixChars,dtype=np.uint8).reshape(len(pixChars)/bpp,bpp)
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(layer.height, layer.width, bpp)


def createResultLayer(image, name, result):
    rlBytes = np.uint8(result).tobytes();
    rl = gimp.Layer(image, name, image.width, image.height, 1, 100,
                    NORMAL_MODE)  # image.active_layer.type  or  RGB_IMAGE
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()


def getnewalpha(image, mask, cFlag):
    if image.shape[2] == 4:  # get rid of alpha channel
        image = image[:, :, 0:3]
    if mask.shape[2] == 4:  # get rid of alpha channel
        mask = mask[:, :, 0:3]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    trimap = mask[:, :, 0]

    cudaFlag = False
    if torch.cuda.is_available() and not cFlag:
        cudaFlag = True

    args = Namespace(crop_or_resize='whole', cuda=cudaFlag, max_size=1600,
                     resume=baseLoc + 'weights/deepmatting/stage1_sad_57.1.pth', stage=1)
    model = net.VGG16(args)

    if cudaFlag:
        ckpt = torch.load(args.resume)
    else:
        ckpt = torch.load(args.resume, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt['state_dict'], strict=True)
    if cudaFlag:
        model = model.cuda()

    # ckpt = torch.load(args.resume)
    # model.load_state_dict(ckpt['state_dict'], strict=True)
    # model = model.cuda()

    torch.cuda.empty_cache()
    with torch.no_grad():
        pred_mattes = inference_img_whole(args, model, image, trimap)
    pred_mattes = (pred_mattes * 255).astype(np.uint8)
    pred_mattes[trimap == 255] = 255
    pred_mattes[trimap == 0] = 0
    # pred_mattes = np.repeat(pred_mattes[:, :, np.newaxis], 3, axis=2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred_mattes = np.dstack((image, pred_mattes))
    return pred_mattes


def deepmatting(imggimp, curlayer, layeri, layerm, cFlag):
    img = channelData(layeri)
    mask = channelData(layerm)
    if img.shape[0] != imggimp.height or img.shape[1] != imggimp.width or mask.shape[0] != imggimp.height or mask.shape[1] != imggimp.width:
        pdb.gimp_message(" Do (Layer -> Layer to Image Size) for both layers and try again.")
    else:
        if torch.cuda.is_available() and not cFlag:
            gimp.progress_init("(Using GPU) Running deep-matting for " + layeri.name + "...")
        else:
            gimp.progress_init("(Using CPU) Running deep-matting for " + layeri.name + "...")
        cpy = getnewalpha(img, mask, cFlag)
        createResultLayer(imggimp, 'new_output', cpy)


register(
    "deep-matting",
    "deep-matting",
    "Running image matting.",
    "Kritik Soman",
    "Your",
    "2020",
    "deepmatting...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [(PF_IMAGE, "image", "Input image", None),
     (PF_DRAWABLE, "drawable", "Input drawable", None),
     (PF_LAYER, "drawinglayer", "Original Image:", None),
     (PF_LAYER, "drawinglayer", "Trimap Mask:", None),
     (PF_BOOL, "fcpu", "Force CPU", False)
     ],
    [],
    deepmatting, menu="<Image>/Layer/GIML-ML")

main()
