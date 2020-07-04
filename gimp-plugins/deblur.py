#!/usr/bin/env python2
import os 
baseLoc = os.path.dirname(os.path.realpath(__file__))+'/'


from gimpfu import *
import sys
sys.path.extend([baseLoc+'gimpenv/lib/python2.7',baseLoc+'gimpenv/lib/python2.7/site-packages',baseLoc+'gimpenv/lib/python2.7/site-packages/setuptools',baseLoc+'DeblurGANv2'])

import cv2
from predictorClass import Predictor
import numpy as np

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

def getdeblur(img):
    predictor = Predictor(weights_path=baseLoc+'DeblurGANv2/'+'best_fpn.h5')
    pred = predictor(img, None)
    return pred

def deblur(img, layer):
    gimp.progress_init("Running for " + layer.name + "...")
    imgmat = channelData(layer)
    pred = getdeblur(imgmat)
    createResultLayer(img,'deblur_'+layer.name,pred)


register(
    "deblur",
    "deblur",
    "Running deblurring.",
    "Kritik Soman",
    "Your",
    "2020",
    "deblur...",
    "*",      # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [   (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    deblur, menu="<Image>/Layer/GIML-ML")

main()
