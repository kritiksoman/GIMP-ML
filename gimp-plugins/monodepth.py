#!/usr/bin/env python2
import os 
baseLoc = os.path.dirname(os.path.realpath(__file__))+'/'


from gimpfu import *
import sys
sys.path.extend([baseLoc+'gimpenv/lib/python2.7',baseLoc+'gimpenv/lib/python2.7/site-packages',baseLoc+'gimpenv/lib/python2.7/site-packages/setuptools',baseLoc+'monodepth2'])


import PIL.Image as pil
import networks
import torch
from torchvision import transforms
import os
import numpy as np
import cv2
# import matplotlib as mpl
# import matplotlib.cm as cm

def getMonoDepth(input_image):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    loc=baseLoc+'monodepth2/'

    model_path = os.path.join(loc+"models", 'mono+stereo_640x192')
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    with torch.no_grad():
        input_image = pil.fromarray(input_image)
        # input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        vmin = disp_resized_np.min()
        disp_resized_np = vmin + (disp_resized_np - vmin) * (vmax - vmin) / (disp_resized_np.max() - vmin)
        disp_resized_np = (255 * (disp_resized_np - vmin) / (vmax - vmin)).astype(np.uint8)
        colormapped_im = cv2.applyColorMap(disp_resized_np, cv2.COLORMAP_HOT)
        colormapped_im = cv2.cvtColor(colormapped_im, cv2.COLOR_BGR2RGB)
        # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im

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

def MonoDepth(img, layer) :
    gimp.progress_init("Generating disparity map for " + layer.name + "...")

    imgmat = channelData(layer)
    cpy=getMonoDepth(imgmat)

    createResultLayer(img,'new_output',cpy)


    

register(
    "MonoDepth",
    "MonoDepth",
    "Generate monocular disparity map based on deep learning.",
    "Kritik Soman",
    "Your",
    "2020",
    "MonoDepth...",
    "*",      # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [   (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    MonoDepth, menu="<Image>/Layer/GIML-ML")

main()
