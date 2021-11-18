import os

baseLoc = os.path.dirname(os.path.realpath(__file__)) + '/'

from gimpfu import *
import sys

activate_this = os.path.join(baseLoc, 'gimpenv', 'bin', 'activate_this.py')
with open(activate_this) as f:
    code = compile(f.read(), activate_this, 'exec')
    exec(code, dict(__file__=activate_this))
sys.path.extend([baseLoc + 'pytorch-SRResNet'])

from argparse import Namespace
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import cv2


def getlabelmat(mask, idx):
    x = np.zeros((mask.shape[0], mask.shape[1], 3))
    x[mask == idx, 0] = colors[idx][0]
    x[mask == idx, 1] = colors[idx][1]
    x[mask == idx, 2] = colors[idx][2]
    return x


def colorMask(mask):
    x = np.zeros((mask.shape[0], mask.shape[1], 3))
    for idx in range(19):
        x = x + getlabelmat(mask, idx)
    return np.uint8(x)


def getnewimg(input_image, s, cFlag, fFlag):
    opt = Namespace(cuda=torch.cuda.is_available() and not cFlag,
                    model=baseLoc + 'weights/super_resolution/model_srresnet.pth',
                    dataset='Set5', scale=s, gpus=0)

    w, h = input_image.shape[0:2]
    cuda = opt.cuda

    if cuda:
        model = torch.load(opt.model)["model"]
    else:
        model = torch.load(opt.model, map_location=torch.device('cpu'))["model"]

    im_input = input_image.astype(np.float32).transpose(2, 0, 1)
    im_input = im_input.reshape(1, im_input.shape[0], im_input.shape[1], im_input.shape[2])
    im_input = Variable(torch.from_numpy(im_input / 255.).float())

    if cuda and not cFlag:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    if fFlag:
        im_h = np.zeros([4 * w, 4 * h, 3])
        wbin = 300
        i = 0
        idx = 0
        t = float(w * h) / float(wbin * wbin)
        while i < w:
            i_end = min(i + wbin, w)
            j = 0
            while j < h:
                j_end = min(j + wbin, h)
                patch = im_input[:, :, i:i_end, j:j_end]
                # patch_merge_out_numpy = denoiser(patch, c, pss, model, model_est, opt, cFlag)
                HR_4x = model(patch)
                HR_4x = HR_4x.cpu().data[0].numpy().astype(np.float32) * 255.
                HR_4x = np.clip(HR_4x, 0., 255.).transpose(1, 2, 0).astype(np.uint8)

                im_h[4 * i:4 * i_end, 4 * j:4 * j_end, :] = HR_4x
                j = j_end
                idx = idx + 1
                gimp.progress_update(float(idx) / float(t))
                gimp.displays_flush()
            i = i_end
    else:
        HR_4x = model(im_input)
        HR_4x = HR_4x.cpu()
        im_h = HR_4x.data[0].numpy().astype(np.float32)
        im_h = im_h * 255.
        im_h = np.clip(im_h, 0., 255.)
        im_h = im_h.transpose(1, 2, 0).astype(np.uint8)
    return im_h


def channelData(layer):  # convert gimp image to numpy
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    bpp = region.bpp
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(layer.height, layer.width, bpp)


def createResultFile(name, layer_np):
    h, w, d = layer_np.shape
    img = pdb.gimp_image_new(w, h, RGB)
    display = pdb.gimp_display_new(img)

    rlBytes = np.uint8(layer_np).tobytes();
    rl = gimp.Layer(img, name, img.width, img.height, RGB, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes

    pdb.gimp_image_insert_layer(img, rl, None, 0)

    gimp.displays_flush()

def createResultLayer(image, name, result):
    rlBytes = np.uint8(result).tobytes();
    rl = gimp.Layer(image, name, image.width, image.height, 0, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()

def super_resolution(img, layer, scale, cFlag, fFlag):
    imgmat = channelData(layer)
    if imgmat.shape[0] != img.height or imgmat.shape[1] != img.width:
        pdb.gimp_message(" Do (Layer -> Layer to Image Size) first and try again.")
    else:
        if torch.cuda.is_available() and not cFlag:
            gimp.progress_init("(Using GPU) Running super-resolution for " + layer.name + "...")
        else:
            gimp.progress_init("(Using CPU) Running  super-resolution for " + layer.name + "...")

        if imgmat.shape[2] == 4:  # get rid of alpha channel
            imgmat = imgmat[:, :, 0:3]
        cpy = getnewimg(imgmat, scale, cFlag, fFlag)
        cpy = cv2.resize(cpy, (0, 0), fx=scale / 4, fy=scale / 4)
        if scale==1:
            createResultLayer(img, layer.name + '_super', cpy)
        else:
            createResultFile(layer.name + '_super', cpy)


register(
    "super-resolution",
    "super-resolution",
    "Running super-resolution.",
    "Kritik Soman",
    "Your",
    "2020",
    "super-resolution...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [(PF_IMAGE, "image", "Input image", None),
     (PF_DRAWABLE, "drawable", "Input drawable", None),
     (PF_SLIDER, "Scale", "Scale", 4, (1, 4, 0.5)),
     (PF_BOOL, "fcpu", "Force CPU", False),
     (PF_BOOL, "ffilter", "Use as filter", True)
     ],
    [],
    super_resolution, menu="<Image>/Layer/GIML-ML")

main()
