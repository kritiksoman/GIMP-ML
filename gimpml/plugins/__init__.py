import pickle
import os
import sys
import cv2

plugin_loc = os.path.dirname(os.path.realpath(__file__)) + "/"
base_loc = os.path.expanduser("~") + "/GIMP-ML/"
# base_loc = "D:/PycharmProjects/"
sys.path.extend([plugin_loc + "MiDaS"])
# data_path = "D:/PycharmProjects/GIMP3-ML-pip/gimpml/"

from mono_run import run_depth
from monodepth_net import MonoDepthNet
import MiDaS_utils as MiDaS_utils
import numpy as np
import cv2
import torch


def get_mono_depth(input_image, cFlag=False):
    image = input_image / 255.0
    out = run_depth(
        image,
        base_loc + "weights/MiDaS/model.pt",
        MonoDepthNet,
        MiDaS_utils,
        target_w=640,
        f=cFlag,
    )
    out = np.repeat(out[:, :, np.newaxis], 3, axis=2)
    d1, d2 = input_image.shape[:2]
    out = cv2.resize(out, (d2, d1))
    # cv2.imwrite("/Users/kritiksoman/PycharmProjects/new/out.png", out)
    return out


if __name__ == "__main__":
    # # This will run when script is run as sub-process
    # dbfile = open(data_path + "data_input", 'rb')
    # data_input = pickle.load(dbfile)
    # dbfile.close()
    # # print(data)
    # data_output = {'args_input': {'processed': 1}, 'image_output': get_mono_depth(data_input['image'])}
    #
    # dbfile = open(data_path + "data_output", 'ab')
    # pickle.dump(data_output, dbfile)  # source, destination
    # dbfile.close()

    image = cv2.imread(os.path.join(base_loc, "cache.png"))[:, :, ::-1]
    output = get_mono_depth(image)
    cv2.imwrite(os.path.join(base_loc, "cache.png"), output[:, :, ::-1])
