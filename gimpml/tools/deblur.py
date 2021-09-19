import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "DeblurGANv2")
sys.path.extend([plugin_loc])

import cv2
from predictorClass import Predictor
import torch
from gimpml.tools.tools_utils import get_weight_path
import traceback


def get_deblur(img, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    predictor = Predictor(
        weights_path=os.path.join(weight_path, "deblur", "best_fpn.h5"), cf=cpu_flag
    )
    if img.shape[2] == 4:  # get rid of alpha channel
        img = img[:, :, 0:3]
    with torch.no_grad():
        pred = predictor(img, None, cf=cpu_flag)
    return pred


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    image = cv2.imread(os.path.join(weight_path, "..", "cache.png"))[:, :, ::-1]
    try:
        output = get_deblur(image, cpu_flag=force_cpu, weight_path=weight_path)
        cv2.imwrite(os.path.join(weight_path, "..", "cache.png"), output[:, :, ::-1])
        with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "wb") as file:
            pickle.dump({"inference_status": "success", "force_cpu": force_cpu}, file)

        # Remove old temporary error files that were saved
        my_dir = os.path.join(weight_path, "..")
        for f_name in os.listdir(my_dir):
            if f_name.startswith("error_log"):
                os.remove(os.path.join(my_dir, f_name))

    except Exception as error:
        with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "wb") as file:
            pickle.dump({"inference_status": "failed"}, file)
        with open(os.path.join(weight_path, "..", "error_log.txt"), "w") as file:
            e_type, e_val, e_tb = sys.exc_info()
            traceback.print_exception(e_type, e_val, e_tb, file=file)
