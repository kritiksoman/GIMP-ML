import os

baseLoc = os.path.dirname(os.path.realpath(__file__)) + '/'
savePath = '/'.join(baseLoc.split('/')[:-2]) + '/output/interpolateframes'
from gimpfu import *
import sys

activate_this = os.path.join(baseLoc, 'gimpenv', 'bin', 'activate_this.py')
with open(activate_this) as f:
    code = compile(f.read(), activate_this, 'exec')
    exec(code, dict(__file__=activate_this))
sys.path.extend([baseLoc + 'RIFE'])

import cv2
import torch
from torch.nn import functional as F
from model import RIFE
import numpy as np



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


def getinter(img_s, img_e, c_flag, string_path):
    exp = 4
    out_path = string_path

    model = RIFE.Model(c_flag)
    model.load_model(baseLoc + 'weights' + '/interpolateframes')
    model.eval()
    model.device(c_flag)

    img0 = img_s
    img1 = img_e

    img0 = (torch.tensor(img0.transpose(2, 0, 1)) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1))/ 255.).unsqueeze(0)
    if torch.cuda.is_available() and not c_flag:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    img0 = img0.to(device)
    img1 = img1.to(device)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    img_list = [img0, img1]
    idx=0
    t=exp * (len(img_list) - 1)
    for i in range(exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        idx=idx+1
        gimp.progress_update(float(idx)/float(t))
        gimp.displays_flush()
        tmp.append(img1)
        img_list = tmp

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(img_list)):
        cv2.imwrite(out_path + '/img{}.png'.format(i),
                    (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w, ::-1])




def interpolateframes(imggimp, curlayer, string_path, layer_s, layer_e, c_flag):
    layer_1 = channelData(layer_s)
    layer_2 = channelData(layer_e)

    if layer_1.shape[0] != imggimp.height or layer_1.shape[1] != imggimp.width or layer_2.shape[0] != imggimp.height or layer_2.shape[1] != imggimp.width:
        pdb.gimp_message(" Do (Layer -> Layer to Image Size) for both layers and try again.")
    else:
        if torch.cuda.is_available() and not c_flag:
            gimp.progress_init("(Using GPU) Running slomo and saving frames in "+string_path)
            # device = torch.device("cuda")
        else:
            gimp.progress_init("(Using CPU) Running slomo and saving frames in "+string_path)
            # device = torch.device("cpu")

        if layer_1.shape[2] == 4:  # get rid of alpha channel
            layer_1 = layer_1[:, :, 0:3]
        if layer_2.shape[2] == 4:  # get rid of alpha channel
            layer_2 = layer_2[:, :, 0:3]
        getinter(layer_1, layer_2, c_flag, string_path)
        # pdb.gimp_message("Saved")


register(
    "interpolate-frames",
    "interpolate-frames",
    "Running slomo...",
    "Kritik Soman",
    "Your",
    "2020",
    "interpolate-frames...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [(PF_IMAGE, "image", "Input image", None),
     (PF_DRAWABLE, "drawable", "Input drawable", None),
     (PF_STRING, "string", "Output Location", savePath),
     (PF_LAYER, "drawinglayer", "Start Frame:", None),
     (PF_LAYER, "drawinglayer", "End Frame:", None),
     (PF_BOOL, "fcpu", "Force CPU", False)
     ],
    [],
    interpolateframes, menu="<Image>/Layer/GIML-ML")

main()
