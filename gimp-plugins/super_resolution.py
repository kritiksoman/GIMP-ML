
from gimpfu import *
import sys
sys.path.extend([baseLoc+'gimpenv/lib/python2.7',baseLoc+'gimpenv/lib/python2.7/site-packages',baseLoc+'gimpenv/lib/python2.7/site-packages/setuptools',baseLoc+'pytorch-SRResNet'])


from argparse import Namespace
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image


def getlabelmat(mask,idx):
    x=np.zeros((mask.shape[0],mask.shape[1],3))
    x[mask==idx,0]=colors[idx][0] 
    x[mask==idx,1]=colors[idx][1] 
    x[mask==idx,2]=colors[idx][2]
    return x 

def colorMask(mask):
    x=np.zeros((mask.shape[0],mask.shape[1],3))
    for idx in range(19):
        x=x+getlabelmat(mask,idx)
    return np.uint8(x)


def getnewimg(input_image,s):
    opt=Namespace(cuda=torch.cuda.is_available(),
        model=baseLoc+'pytorch-SRResNet/model/model_srresnet.pth',
        dataset='Set5',scale=s,gpus=0)

    im_l=Image.fromarray(input_image)
    cuda = opt.cuda

    if cuda:
        model = torch.load(opt.model)["model"]
    else:
        model = torch.load(opt.model,map_location=torch.device('cpu'))["model"]

    im_l=np.array(im_l)
    im_l = im_l.astype(float)
    im_input = im_l.astype(np.float32).transpose(2,0,1)
    im_input = im_input.reshape(1,im_input.shape[0],im_input.shape[1],im_input.shape[2])
    im_input = Variable(torch.from_numpy(im_input/255.).float())

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    HR_4x = model(im_input)

    HR_4x = HR_4x.cpu()

    im_h = HR_4x.data[0].numpy().astype(np.float32)

    im_h = im_h*255.
    im_h = np.clip(im_h, 0., 255.)
    im_h = im_h.transpose(1,2,0).astype(np.uint8)

    return im_h


def channelData(layer):#convert gimp image to numpy
    region=layer.get_pixel_rgn(0, 0, layer.width,layer.height)
    pixChars=region[:,:] # Take whole layer
    bpp=region.bpp
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
    

def super_resolution(img, layer,scale) :
    if torch.cuda.is_available():
        gimp.progress_init("(Using GPU) Running for " + layer.name + "...")
    else:
        gimp.progress_init("(Using CPU) Running for " + layer.name + "...")

    imgmat = channelData(layer)
    cpy = getnewimg(imgmat,scale)
    genNewImg(layer.name+'_upscaled',cpy)
    

register(
    "super-resolution",
    "super-resolution",
    "Running super-resolution.",
    "Kritik Soman",
    "Your",
    "2020",
    "super-resolution...",
    "*",      # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [   (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
        (PF_SLIDER, "Scale",  "Scale", 4, (1.1, 4, 0.5))
    ],
    [],
    super_resolution, menu="<Image>/Layer/GIML-ML")

main()
