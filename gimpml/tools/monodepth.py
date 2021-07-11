import pickle
import os
import sys

plugin_loc = os.path.dirname(os.path.realpath(__file__)) + '/'
sys.path.extend([plugin_loc + 'MiDaS'])

from mono_run import run_depth
from monodepth_net import MonoDepthNet
import MiDaS_utils as MiDaS_utils
import numpy as np
import cv2
import torch


def get_weight_path():
    config_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(config_path, 'gimp_ml_config.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    weight_path = data_output["weight_path"]
    return weight_path


def get_mono_depth(input_image, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    image = input_image / 255.0
    with torch.no_grad():
        out = run_depth(image, os.path.join(weight_path, 'MiDaS', 'model.pt'), MonoDepthNet, MiDaS_utils, target_w=640,
                        f=cpu_flag)
    out = np.repeat(out[:, :, np.newaxis], 3, axis=2)
    d1, d2 = input_image.shape[:2]
    out = cv2.resize(out, (d2, d1))
    return out


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    image = cv2.imread(os.path.join(weight_path, '..', "cache.png"))[:, :, ::-1]
    force_cpu = data_output["force_cpu"]
    try:
        output = get_mono_depth(image, cpu_flag=force_cpu, weight_path=weight_path)
        cv2.imwrite(os.path.join(weight_path, '..', 'cache.png'), output[:, :, ::-1])
        with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'wb') as file:
            pickle.dump({"inference_status": "success", "force_cpu": force_cpu}, file)

        # Remove old temporary error files that were saved
        my_dir = os.path.join(weight_path, '..')
        for f_name in os.listdir(my_dir):
            if f_name.startswith("error_log"):
                os.remove(os.path.join(my_dir, f_name))
    except Exception as error:
        with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'wb') as file:
            pickle.dump({"inference_status": "failed"}, file)
        with open(os.path.join(weight_path, '..', 'error_log.txt'), 'w') as file:
            file.write(str(error))
