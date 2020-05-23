import os
import sys

from _util import add_gimpenv_to_pythonpath, baseLoc

add_gimpenv_to_pythonpath()
modelDir = os.path.join(baseLoc, 'neural-colorization')
sys.path.append(modelDir)

from gimpfu import *
import torch
from model import generator
from torch.autograd import Variable
from scipy.ndimage import zoom
from argparse import Namespace
import numpy as np
import cv2

def getcolor(input_image):
    p = np.repeat(input_image, 3, axis=2)

    if torch.cuda.is_available():
        g_available=1
    else:
        g_available=-1

    args=Namespace(model=os.path.join(modelDir, 'model.pth'), gpu=g_available)

    G = generator()

    if torch.cuda.is_available():
        G=G.cuda()
        G.load_state_dict(torch.load(args.model))
    else:
        G.load_state_dict(torch.load(args.model,map_location=torch.device('cpu')))

    p = p.astype(np.float32)
    p = p / 255
    img_yuv = cv2.cvtColor(p, cv2.COLOR_RGB2YUV)
    # img_yuv = rgb2yuv(p)
    H,W,_ = img_yuv.shape
    infimg = np.expand_dims(np.expand_dims(img_yuv[...,0], axis=0), axis=0)
    img_variable = Variable(torch.Tensor(infimg-0.5))
    if args.gpu>=0:
        img_variable=img_variable.cuda(args.gpu)
    res = G(img_variable)
    uv=res.cpu().detach().numpy()
    uv[:,0,:,:] *= 0.436
    uv[:,1,:,:] *= 0.615
    (_,_,H1,W1) = uv.shape
    uv = zoom(uv,(1,1,float(H)/H1,float(W)/W1))
    yuv = np.concatenate([infimg,uv],axis=1)[0]
    # rgb=yuv2rgb(yuv.transpose(1,2,0))
    # out=(rgb.clip(min=0,max=1)*255)[:,:,[0,1,2]]
    rgb = cv2.cvtColor(yuv.transpose(1, 2, 0)*255, cv2.COLOR_YUV2RGB)
    rgb = rgb.clip(min=0,max=255)
    out = rgb.astype(np.uint8)

    return out

def channelData(layer):#convert gimp image to numpy
    region=layer.get_pixel_rgn(0, 0, layer.width,layer.height)
    pixChars=region[:,:] # Take whole layer
    bpp=region.bpp
    # return np.frombuffer(pixChars,dtype=np.uint8).reshape(len(pixChars)/bpp,bpp)
    return np.frombuffer(pixChars,dtype=np.uint8).reshape(layer.height,layer.width,bpp)

def createResultLayer(image,name,result):
    rlBytes=np.uint8(result).tobytes();
    rl=gimp.Layer(image,name,image.width,image.height,image.active_layer.type,100,NORMAL_MODE)
    region=rl.get_pixel_rgn(0, 0, rl.width,rl.height,True)
    region[:,:]=rlBytes
    image.add_layer(rl,0)
    gimp.displays_flush()

def genNewImg(name,layer_np):
    h,w,d=layer_np.shape
    img=pdb.gimp_image_new(w, h, RGB)
    display=pdb.gimp_display_new(img)

    rlBytes=np.uint8(layer_np).tobytes();
    rl=gimp.Layer(img,name,img.width,img.height,RGB,100,NORMAL_MODE)
    region=rl.get_pixel_rgn(0, 0, rl.width,rl.height,True)
    region[:,:]=rlBytes

    pdb.gimp_image_insert_layer(img, rl, None, 0)
    gimp.displays_flush()

def colorize(img, layer) :
    gimp.progress_init("Coloring " + layer.name + "...")

    imgmat = channelData(layer)
    cpy=getcolor(imgmat)

    genNewImg(layer.name+'_colored',cpy)

    

register(
    "colorize",
    "colorize",
    "Generate monocular disparity map based on deep learning.",
    "Kritik Soman",
    "Your",
    "2020",
    "colorize...",
    "*",      # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [   (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    colorize, menu="<Image>/Layer/GIML-ML")

main()
