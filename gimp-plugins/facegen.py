import os
import sys

from _util import add_gimpenv_to_pythonpath, baseLoc

add_gimpenv_to_pythonpath()
modelDir = os.path.join(baseLoc, 'CelebAMask-HQ/MaskGAN_demo')
sys.path.append(modelDir)

from gimpfu import *
import torch
from argparse import Namespace
from models.models import create_model
from data.base_dataset import get_params, get_transform
import os
import numpy as np
from PIL import Image

colors = np.array([[0, 0, 0], [204, 0, 0], [76, 153, 0], \
[204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], \
[51, 255, 255], [102, 51, 0], [255, 0, 0], [102, 204, 0], \
[255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], \
[0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]])
colors = colors.astype(np.uint8)

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
    # mask=np.dstack((mask1,mask2,mask3))
    return np.uint8(x)

def labelMask(mask):
    x=np.zeros((mask.shape[0],mask.shape[1],3))
    for idx in range(19):
        tmp=np.logical_and(mask[:,:,0]==colors[idx][0],mask[:,:,1]==colors[idx][1])
        tmp2=np.logical_and(tmp,mask[:,:,2]==colors[idx][2])
        x[tmp2]=idx
    return x     

def getOptions():
    mydict={'aspect_ratio': 1.0,
    'batchSize': 1,
    'checkpoints_dir': os.path.join(modelDir, 'checkpoints'),
    'cluster_path': 'features_clustered_010.npy',
    'data_type': 32,
    'dataroot': '../Data_preprocessing/',
    'display_winsize': 512,
    'engine': None,
    'export_onnx': None,
    'fineSize': 512,
    'gpu_ids': [0],
    'how_many': 1000,
    'input_nc': 3,
    'isTrain': False,
    'label_nc': 19,
    'loadSize': 512,
    'max_dataset_size': 'inf',
    'model': 'pix2pixHD',
    'nThreads': 2,
    'n_blocks_global': 4,
    'n_blocks_local': 3,
    'n_downsample_global': 4,
    'n_local_enhancers': 1,
    'name': 'label2face_512p',
    'netG': 'global',
    'ngf': 64,
    'niter_fix_global': 0,
    'no_flip': False,
    'norm': 'instance',
    'ntest': 'inf',
    'onnx': None,
    'output_nc': 3,
    'phase': 'test',
    'resize_or_crop': 'scale_width',
    'results_dir': './results/',
    'serial_batches': False,
    'tf_log': False,
    'use_dropout': False,
    'use_encoded_image': False,
    'verbose': False,
    'which_epoch': 'latest'}
    args = Namespace(**mydict)
    return args

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

def getnewface(img,mask,mask_m):
    h,w,d = img.shape
    img = Image.fromarray(img)
    lmask = labelMask(mask)
    lmask_m = labelMask(mask_m)


    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    opt = getOptions()

    model = create_model(opt)   

    params = get_params(opt, (512,512))
    transform_mask = get_transform(opt, params, method=Image.NEAREST, normalize=False, normalize_mask=True)
    transform_image = get_transform(opt, params)
    mask = transform_mask(Image.fromarray(np.uint8(lmask))) 
    mask_m = transform_mask(Image.fromarray(np.uint8(lmask_m)))
    img = transform_image(img)
 
    generated = model.inference(torch.FloatTensor([mask_m.numpy()]), torch.FloatTensor([mask.numpy()]), torch.FloatTensor([img.numpy()]))   

    result = generated.permute(0, 2, 3, 1)
    if torch.cuda.is_available():
        result = result.cpu().numpy()
    else:
        result = result.detach().numpy()

    result = (result + 1) * 127.5
    result = np.asarray(result[0,:,:,:], dtype=np.uint8)
    result = Image.fromarray(result)
    result = result.resize([w,h])
    
    result = np.array(result)
    return result


def facegen(imggimp, curlayer,layeri,layerm,layermm) :
    if torch.cuda.is_available():
        gimp.progress_init("(Using GPU) Running face gen for " + layeri.name + "...")
    else:
        gimp.progress_init("(Using CPU) Running face gen for " + layeri.name + "...")

    img = channelData(layeri)
    mask = channelData(layerm)
    mask_m = channelData(layermm)

    cpy=getnewface(img,mask,mask_m)
    createResultLayer(imggimp,'new_output',cpy)

    

register(
    "facegen",
    "facegen",
    "Running face gen.",
    "Kritik Soman",
    "Your",
    "2020",
    "facegen...",
    "*",      # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [   (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
        (PF_LAYER, "drawinglayer", "Original Image:", None),
        (PF_LAYER, "drawinglayer", "Original Mask:", None),
        (PF_LAYER, "drawinglayer", "Modified Mask:", None),
    ],
    [],
    facegen, menu="<Image>/Layer/GIML-ML")

main()
