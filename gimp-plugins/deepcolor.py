import os 
baseLoc = os.path.dirname(os.path.realpath(__file__))+'/'
from gimpfu import *
import sys
sys.path.extend([baseLoc+'gimpenv/lib/python2.7',baseLoc+'gimpenv/lib/python2.7/site-packages',baseLoc+'gimpenv/lib/python2.7/site-packages/setuptools',baseLoc+'ideepcolor'])
import numpy as np
import torch
import cv2
from data import colorize_image as CI

def createResultLayer(image,name,result):
    rlBytes=np.uint8(result).tobytes();
    rl=gimp.Layer(image,name,image.width,image.height)
    # ,image.active_layer.type,100,NORMAL_MODE)
    region=rl.get_pixel_rgn(0, 0, rl.width,rl.height,True)
    region[:,:]=rlBytes
    image.add_layer(rl,0)
    gimp.displays_flush()

def channelData(layer):#convert gimp image to numpy
    region=layer.get_pixel_rgn(0, 0, layer.width,layer.height)
    pixChars=region[:,:] # Take whole layer
    bpp=region.bpp
    return np.frombuffer(pixChars,dtype=np.uint8).reshape(layer.height,layer.width,bpp)


def deepcolor(tmp1, tmp2, ilayerimg,ilayerc,cflag) :
    layerimg = channelData(ilayerimg)
    layerc = channelData(ilayerc)

    if ilayerimg.name == ilayerc.name: # if local color hints are not provided by user
            mask = np.zeros((1, 256, 256))  # giving no user points, so mask is all 0's
            input_ab = np.zeros((2, 256, 256))  # ab values of user points, default to 0 for no input
    else:
        if layerc.shape[2] == 3:  # error
            pdb.gimp_message("Alpha channel missing in " + ilayerc.name + " !")
            return
        else:
            input_ab = cv2.cvtColor(layerc[:,:,0:3].astype(np.float32)/255, cv2.COLOR_RGB2LAB)
            mask = layerc[:,:,3]>0
            mask = mask.astype(np.uint8)
            input_ab = cv2.resize(input_ab,(256,256))
            mask = cv2.resize(mask, (256, 256))
            mask = mask[np.newaxis, :, :]
            input_ab = input_ab[:,:, 1:3].transpose((2, 0, 1))

    if layerimg.shape[2] == 4: #remove alpha channel in image if present
        layerimg = layerimg[:,:,0:3]

    if torch.cuda.is_available() and not cflag:
        gimp.progress_init("(Using GPU) Running deepcolor for " + ilayerimg.name + "...")
        gpu_id = 0
    else:
        gimp.progress_init("(Using CPU) Running deepcolor for " + ilayerimg.name + "...")
        gpu_id = None

    colorModel = CI.ColorizeImageTorch(Xd=256)
    colorModel.prep_net(gpu_id, baseLoc + 'weights/colorize/caffemodel.pth')
    colorModel.load_image(layerimg)  # load an image

    img_out = colorModel.net_forward(input_ab, mask,f=cflag)  # run model, returns 256x256 image
    img_out_fullres = colorModel.get_img_fullres()  # get image at full resolution

    createResultLayer(tmp1, 'new_' + ilayerimg.name, img_out_fullres)

    

register(
    "deepcolor",
    "deepcolor",
    "Running deepcolor.",
    "Kritik Soman",
    "Your",
    "2020",
    "deepcolor...",
    "*",      # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [   (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
        (PF_LAYER, "drawinglayer", "Original Image:", None),
        (PF_LAYER, "drawinglayer", "Color Mask:", None),
        (PF_BOOL, "fcpu", "Force CPU", False)
    ],
    [],
    deepcolor, menu="<Image>/Layer/GIML-ML")

main()
