import os

baseLoc = os.path.dirname(os.path.realpath(__file__)) + '/'

from gimpfu import *
import sys

sys.path.extend([baseLoc + 'gimpenv/lib/python2.7', baseLoc + 'gimpenv/lib/python2.7/site-packages',
                 baseLoc + 'gimpenv/lib/python2.7/site-packages/setuptools', baseLoc + 'EnlightenGAN'])

from argparse import Namespace
import cv2
import numpy as np
import torch
from models.models import create_model
from data.base_dataset import get_transform




def getEnlighten(input_image,cFlag):

    opt = Namespace(D_P_times2=False, IN_vgg=False, aspect_ratio=1.0, batchSize=1,
                checkpoints_dir=baseLoc+'weights/', dataroot='test_dataset',
                dataset_mode='unaligned', display_id=1, display_port=8097,
                display_single_pane_ncols=0, display_winsize=256, fcn=0,
                fineSize=256, gpu_ids=[0], high_times=400, how_many=50,
                hybrid_loss=False, identity=0.0, input_linear=False, input_nc=3,
                instance_norm=0.0, isTrain=False, l1=10.0, lambda_A=10.0,
                lambda_B=10.0, latent_norm=False, latent_threshold=False,
                lighten=False, linear=False, linear_add=False, loadSize=286,
                low_times=200, max_dataset_size='inf', model='single',
                multiply=False, nThreads=1, n_layers_D=3, n_layers_patchD=3,
                name='enlightening', ndf=64, new_lr=False, ngf=64, no_dropout=True,
                no_flip=True, no_vgg_instance=False, noise=0, norm='instance',
                norm_attention=False, ntest='inf', output_nc=3, patchD=False,
                patchD_3=0, patchSize=64, patch_vgg=False, phase='test',
                resize_or_crop='no', results_dir='./results/', self_attention=True,
                serial_batches=True, skip=1.0, syn_norm=False, tanh=False,
                times_residual=True, use_avgpool=0, use_mse=False, use_norm=1.0,
                use_ragan=False, use_wgan=0.0, vary=1, vgg=0, vgg_choose='relu5_3',
                vgg_maxpooling=False, vgg_mean=False, which_direction='AtoB',
                which_epoch='200', which_model_netD='basic', which_model_netG='sid_unet_resize', cFlag=cFlag)

    im = cv2.cvtColor(input_image,cv2.COLOR_RGB2BGR)
    transform = get_transform(opt)
    A_img = transform(im)
    r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
    A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
    A_gray = torch.unsqueeze(A_gray, 0)
    data = {'A': A_img.unsqueeze(0), 'B': A_img.unsqueeze(0), 'A_gray': A_gray.unsqueeze(0), 'input_img': A_img.unsqueeze(0), 'A_paths': 'A_path', 'B_paths': 'B_path'}

    model = create_model(opt)
    model.set_input(data)
    visuals = model.predict()
    out = visuals['fake_B'].astype(np.uint8)
    out = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)
    # cv2.imwrite("/Users/kritiksoman/PycharmProjects/new/out.png", out)
    return out


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


def Enlighten(img, layer,cFlag):
    imgmat = channelData(layer)
    if imgmat.shape[0] != img.height or imgmat.shape[1] != img.width:
        pdb.gimp_message(" Do (Layer -> Layer to Image Size) first and try again.")
    else:
        if torch.cuda.is_available() and not cFlag:
            gimp.progress_init("(Using GPU) Enlighten " + layer.name + "...")
        else:
            gimp.progress_init("(Using CPU) Enlighten " + layer.name + "...")

        if imgmat.shape[2] == 4:  # get rid of alpha channel
            imgmat = imgmat[:,:,0:3]
        cpy = getEnlighten(imgmat,cFlag)
        createResultLayer(img, 'new_output', cpy)


register(
    "enlighten",
    "enlighten",
    "Enlighten image based on deep learning.",
    "Kritik Soman",
    "Your",
    "2020",
    "enlighten...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [(PF_IMAGE, "image", "Input image", None),
     (PF_DRAWABLE, "drawable", "Input drawable", None),
     (PF_BOOL, "fcpu", "Force CPU", False)
     ],
    [],
    Enlighten, menu="<Image>/Layer/GIML-ML")

main()