import pickle
import os
import sys

plugin_loc = os.path.dirname(os.path.realpath(__file__)) + '/'
sys.path.extend([plugin_loc + 'DeblurGANv2'])

import cv2
from predictorClass import Predictor
import numpy as np
import torch


def get_weight_path():
    config_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(config_path, 'gimp_ml_config.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    weight_path = data_output["weight_path"]
    return weight_path


def get_deblur(img, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path=get_weight_path()
    predictor = Predictor(weights_path=os.path.join(weight_path, 'deblur', 'best_fpn.h5'), cf=cpu_flag)
    if img.shape[2] == 4:  # get rid of alpha channel
        img = img[:, :, 0:3]
    pred = predictor(img, None, cf=cpu_flag)
    return pred


if __name__ == "__main__":
    weight_path = get_weight_path()
    image = cv2.imread(os.path.join(weight_path, '..', "cache.png"))[:, :, ::-1]
    with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    output = get_deblur(image, cpu_flag=force_cpu, weight_path=weight_path)
    cv2.imwrite(os.path.join(weight_path, '..', 'cache.png'), output[:, :, ::-1])
    # with open(os.path.join(weight_path, 'gimp_ml_run.pkl'), 'wb') as file:
    #     pickle.dump({"run_success": True}, file)
