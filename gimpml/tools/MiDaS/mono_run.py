"""Compute depth maps for images in the input folder.
"""
# import os
# import glob
import torch

# from monodepth_net import MonoDepthNet
# import utils
# import matplotlib.pyplot as plt
import numpy as np
import cv2

# import imageio


def run_depth(img, model_path, Net, utils, target_w=None, f=False):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    # print("initialize")

    # select device
    # device = torch.device("cpu")
    # print("device: %s" % device)

    # load network
    model = Net(model_path)
    if torch.cuda.is_available() and not f:
        model.cuda()
    model.eval()

    # get input
    # img_names = glob.glob(os.path.join(input_path, "*"))
    # num_images = len(img_names)

    # create output folder
    # os.makedirs(output_path, exist_ok=True)

    # print("start processing")

    # for ind, img_name in enumerate(img_names):

    # print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

    # input
    # img = utils.read_image(img_name)
    w = img.shape[1]
    scale = 640.0 / max(img.shape[0], img.shape[1])
    target_height, target_width = int(round(img.shape[0] * scale)), int(
        round(img.shape[1] * scale)
    )
    img_input = utils.resize_image(img)
    # print(img_input.shape)
    if torch.cuda.is_available() and not f:
        img_input = img_input.cuda()
    # compute
    with torch.no_grad():
        out = model.forward(img_input)

    depth = utils.resize_depth(out, target_width, target_height)
    img = cv2.resize(
        (img * 255).astype(np.uint8),
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    # np.save(filename + '.npy', depth)
    # utils.write_depth(filename, depth, bits=2)
    depth_min = depth.min()
    depth_max = depth.max()
    bits = 1
    max_val = (2 ** (8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0
    out = out.astype("uint8")
    # cv2.imwrite("out.png", out)
    return out
    # print("finished")
