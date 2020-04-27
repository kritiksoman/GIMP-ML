import torch
from model import generator
from torch.autograd import Variable
from scipy.ndimage import zoom
import cv2
import os
from PIL import Image
import argparse
import numpy as np
from skimage.color import rgb2yuv,yuv2rgb

def parse_args():
    parser = argparse.ArgumentParser(description="Colorize images")
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        required=True,
                        help="input image/input dir")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        required=True,
                        help="output image/output dir")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        required=True,
                        help="location for model (Generator)")
    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="which GPU to use? [-1 for cpu]")
    args = parser.parse_args()
    return args

args = parse_args()

G = generator()

if torch.cuda.is_available():
# args.gpu>=0:
    G=G.cuda(args.gpu)
    G.load_state_dict(torch.load(args.model))
else:
    G.load_state_dict(torch.load(args.model,map_location=torch.device('cpu')))

def inference(G,in_path,out_path):
    p=Image.open(in_path).convert('RGB')
    img_yuv = rgb2yuv(p)
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
    uv = zoom(uv,(1,1,H/H1,W/W1))
    yuv = np.concatenate([infimg,uv],axis=1)[0]
    rgb=yuv2rgb(yuv.transpose(1,2,0))
    cv2.imwrite(out_path,(rgb.clip(min=0,max=1)*256)[:,:,[2,1,0]])


if not os.path.isdir(args.input):
    inference(G,args.input,args.output)
else:
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    for f in os.listdir(args.input):
        inference(G,os.path.join(args.input,f),os.path.join(args.output,f))

