import pickle
import os
import sys

plugin_loc = os.path.dirname(os.path.realpath(__file__)) + '/'
sys.path.extend([plugin_loc + 'EnlightenGAN'])

from argparse import Namespace
import cv2
import numpy as np
import torch
from enlighten_models.models import create_model
from enlighten_data.base_dataset import get_transform



def get_enlighten(input_image, cpu_flag=False):
    opt = Namespace(D_P_times2=False, IN_vgg=False, aspect_ratio=1.0, batchSize=1,
                    checkpoints_dir=weight_path, dataroot='test_dataset',
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
                    which_epoch='200', which_model_netD='basic', which_model_netG='sid_unet_resize', cFlag=cpu_flag)

    im = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    transform = get_transform(opt)
    A_img = transform(im)
    r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
    A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
    A_gray = torch.unsqueeze(A_gray, 0)
    data = {'A': A_img.unsqueeze(0), 'B': A_img.unsqueeze(0), 'A_gray': A_gray.unsqueeze(0),
            'input_img': A_img.unsqueeze(0), 'A_paths': 'A_path', 'B_paths': 'B_path'}

    model = create_model(opt)
    model.set_input(data)
    visuals = model.predict()
    out = visuals['fake_B'].astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("/Users/kritiksoman/PycharmProjects/new/out.png", out)
    return out


if __name__ == "__main__":
    config_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(config_path, 'gimp_ml_config.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    weight_path = data_output["weight_path"]
    image = cv2.imread(os.path.join(weight_path, '..', "cache.png"))[:, :, ::-1]
    with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    output = get_enlighten(image, cpu_flag=force_cpu)
    cv2.imwrite(os.path.join(weight_path, '..', 'cache.png'), output[:, :, ::-1])
    # with open(os.path.join(weight_path, 'gimp_ml_run.pkl'), 'wb') as file:
    #     pickle.dump({"run_success": True}, file)
