import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RIFE")
sys.path.extend([plugin_loc])

import cv2
import torch
from torch.nn import functional as F
from rife_model import RIFE
import numpy as np
from gimpml.tools.tools_utils import get_weight_path
import traceback


def get_inter(img_s, img_e, string_path, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    exp = 4
    out_path = string_path

    model = RIFE.Model(cpu_flag)
    model.load_model(os.path.join(weight_path, "interpolateframes"))
    model.eval()
    model.device(cpu_flag)

    img0 = img_s
    img1 = img_e

    img0 = (torch.tensor(img0.transpose(2, 0, 1).copy()) / 255.0).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1).copy()) / 255.0).unsqueeze(0)
    if torch.cuda.is_available() and not cpu_flag:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    img0 = img0.to(device)
    img1 = img1.to(device)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    img_list = [img0, img1]
    for i in range(exp):
        tmp = []
        for j in range(len(img_list) - 1):
            with torch.no_grad():
                mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(img_list)):
        cv2.imwrite(
            os.path.join(out_path, "img{}.png".format(i)),
            (img_list[i][0] * 255)
            .byte()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)[:h, :w, ::-1],
        )


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    gio_file = data_output["gio_file"]
    image1 = cv2.imread(os.path.join(weight_path, "..", "cache0.png"))[:, :, ::-1]
    image2 = cv2.imread(os.path.join(weight_path, "..", "cache1.png"))[:, :, ::-1]
    try:
        get_inter(image1, image2, gio_file, cpu_flag=force_cpu, weight_path=weight_path)
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
