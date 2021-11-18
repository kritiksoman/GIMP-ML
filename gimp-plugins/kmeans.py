import os 
baseLoc = os.path.dirname(os.path.realpath(__file__))+'/'


from gimpfu import *
import sys

activate_this = os.path.join(baseLoc, 'gimpenv', 'bin', 'activate_this.py')
with open(activate_this) as f:
    code = compile(f.read(), activate_this, 'exec')
    exec(code, dict(__file__=activate_this))

import numpy as np
from scipy.cluster.vq import kmeans2


def channelData(layer):#convert gimp image to numpy
    region=layer.get_pixel_rgn(0, 0, layer.width,layer.height)
    pixChars=region[:,:] # Take whole layer
    bpp=region.bpp
    # return np.frombuffer(pixChars,dtype=np.uint8).reshape(len(pixChars)/bpp,bpp)
    return np.frombuffer(pixChars,dtype=np.uint8).reshape(layer.height,layer.width,bpp)


def createResultLayer(image,name,result):
    rlBytes=np.uint8(result).tobytes();
    rl=gimp.Layer(image,name,image.width,image.height,0,100,NORMAL_MODE)#1 is for RGB with alpha
    region=rl.get_pixel_rgn(0, 0, rl.width,rl.height,True)
    region[:,:]=rlBytes
    image.add_layer(rl,0)
    gimp.displays_flush()

def kmeans(imggimp, curlayer,layeri,n_clusters,locflag) :
    image = channelData(layeri)
    if image.shape[0] != imggimp.height or image.shape[1] != imggimp.width:
        pdb.gimp_message(" Do (Layer -> Layer to Image Size) first and try again.")
    else:
        if image.shape[2] == 4:  # get rid of alpha channel
            image = image[:,:,0:3]
        h,w,d = image.shape
        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values = image.reshape((-1, 3))

        if locflag:
            xx,yy = np.meshgrid(range(w),range(h))
            x = xx.reshape(-1,1)
            y = yy.reshape(-1,1)
            pixel_values = np.concatenate((pixel_values,x,y),axis=1)

        pixel_values = np.float32(pixel_values)
        c,out = kmeans2(pixel_values,n_clusters)

        if locflag:
            c = np.uint8(c[:,0:3])
        else:
            c = np.uint8(c)

        segmented_image = c[out.flatten()]
        segmented_image = segmented_image.reshape((h,w,d))
        createResultLayer(imggimp,'new_output',segmented_image)

    
register(
    "kmeans",
    "kmeans clustering",
    "Running kmeans clustering.",
    "Kritik Soman",
    "Your",
    "2020",
    "kmeans...",
    "*",      # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [   (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
        (PF_LAYER, "drawinglayer", "Original Image", None),
        (PF_INT, "depth", "Number of clusters", 3),
        (PF_BOOL, "position", "Use position", False)
          
    ],
    [],
    kmeans, menu="<Image>/Layer/GIML-ML")

main()
