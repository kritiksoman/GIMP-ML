import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ideepcolor")
sys.path.extend([plugin_loc])

import numpy as np
import torch
import cv2
from color_data import colorize_image as CI
from gimpml.tools.tools_utils import get_weight_path
import traceback


def get_deepcolor(layerimg, layerc=None, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    if layerc is not None:
        input_ab = cv2.cvtColor(
            layerc[:, :, 0:3].astype(np.float32) / 255, cv2.COLOR_RGB2LAB
        )
        mask = layerc[:, :, 3] > 0
        input_ab = cv2.resize(input_ab, (256, 256))
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (256, 256))
        mask = mask[np.newaxis, :, :]
        input_ab = input_ab[:, :, 1:3].transpose((2, 0, 1))
    else:
        mask = np.zeros((1, 256, 256))  # giving no user points, so mask is all 0's
        input_ab = np.zeros((2, 256, 256))

    gpu_id = 0 if torch.cuda.is_available() and not cpu_flag else None

    if (
        len(layerimg.shape) == 3 and layerimg.shape[2] == 4
    ):  # remove alpha channel in image if present
        layerimg = layerimg[:, :, 0:3]
    elif len(layerimg.shape) == 2:
        layerimg = np.repeat(layerimg[:, :, np.newaxis], 3, axis=2)

    colorModel = CI.ColorizeImageTorch(Xd=256)
    colorModel.prep_net(gpu_id, os.path.join(weight_path, "colorize", "caffemodel.pth"))
    colorModel.load_image(layerimg)  # load an image

    with torch.no_grad():
        img_out = colorModel.net_forward(
            input_ab, mask, f=cpu_flag
        )  # run model, returns 256x256 image
        img_out_fullres = colorModel.get_img_fullres()  # get image at full resolution
    return img_out_fullres


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    n_drawables = data_output["n_drawables"]
    image1 = cv2.imread(
        os.path.join(weight_path, "..", "cache0.png"), cv2.IMREAD_UNCHANGED
    )
    image2 = None
    if n_drawables == 2:
        image2 = cv2.imread(
            os.path.join(weight_path, "..", "cache1.png"), cv2.IMREAD_UNCHANGED
        )
    force_cpu = data_output["force_cpu"]
    try:
        if n_drawables == 1:
            output = get_deepcolor(image1, cpu_flag=force_cpu, weight_path=weight_path)
        elif (
            image1.shape[2] == 4
            and (np.sum(image1 == [0, 0, 0, 0]))
            / (image1.shape[0] * image1.shape[1] * 4)
            > 0.8
        ):
            image2 = image2[:, :, [2, 1, 0]]
            image1 = image1[:, :, [2, 1, 0, 3]]
            output = get_deepcolor(
                image2, image1, cpu_flag=force_cpu, weight_path=weight_path
            )
        else:
            image1 = image1[:, :, [2, 1, 0]]
            image2 = image2[:, :, [2, 1, 0, 3]]
            output = get_deepcolor(
                image1, image2, cpu_flag=force_cpu, weight_path=weight_path
            )
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
