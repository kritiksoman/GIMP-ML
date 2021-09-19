import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pytorch-deep-image-matting")
sys.path.extend([plugin_loc])

import torch
from argparse import Namespace
import deepmatting_net
import cv2
import os
import numpy as np
from deploy import inference_img_whole
from gimpml.tools.tools_utils import get_weight_path
import traceback


def get_matting(image, mask, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    if image.shape[2] == 4:  # get rid of alpha channel
        image = image[:, :, 0:3]
    if mask.shape[2] == 4:  # get rid of alpha channel
        mask = mask[:, :, 0:3]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    trimap = mask[:, :, 0]

    cuda_flag = False
    if torch.cuda.is_available() and not cpu_flag:
        cuda_flag = True

    args = Namespace(
        crop_or_resize="whole",
        cuda=cuda_flag,
        max_size=1600,
        resume=os.path.join(weight_path, "deepmatting", "stage1_sad_57.1.pth"),
        stage=1,
    )
    model = deepmatting_net.VGG16(args)

    if cuda_flag:
        ckpt = torch.load(args.resume)
    else:
        ckpt = torch.load(args.resume, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    if cuda_flag:
        model = model.cuda()

    torch.cuda.empty_cache()
    with torch.no_grad():
        pred_mattes = inference_img_whole(args, model, image, trimap)
    pred_mattes = (pred_mattes * 255).astype(np.uint8)
    pred_mattes[trimap == 255] = 255
    pred_mattes[trimap == 0] = 0

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pred_mattes = np.dstack((image, pred_mattes))
    return pred_mattes


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    image1 = cv2.imread(os.path.join(weight_path, "..", "cache0.png"))[:, :, ::-1]
    image2 = cv2.imread(os.path.join(weight_path, "..", "cache1.png"))[:, :, ::-1]
    try:
        if (
            np.sum(image1 == [0, 0, 0])
            + np.sum(image1 == [255, 255, 255])
            + np.sum(image1 == [128, 128, 128])
        ) / (image1.shape[0] * image1.shape[1] * 3) > 0.8:
            output = get_matting(
                image2, image1, cpu_flag=force_cpu, weight_path=weight_path
            )
        else:
            output = get_matting(
                image1, image2, cpu_flag=force_cpu, weight_path=weight_path
            )
        cv2.imwrite(
            os.path.join(weight_path, "..", "cache.png"), output[:, :, [2, 1, 0, 3]]
        )
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