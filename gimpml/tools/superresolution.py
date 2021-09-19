import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pytorch-SRResNet")
sys.path.extend([plugin_loc])

from argparse import Namespace
import torch
from torch.autograd import Variable
import numpy as np
import cv2
import warnings
from gimpml.tools.tools_utils import get_weight_path
warnings.filterwarnings("ignore")
import traceback


def get_super(input_image, s=4, cpu_flag=False, fFlag=True, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    opt = Namespace(
        cuda=torch.cuda.is_available() and not cpu_flag,
        model=os.path.join(weight_path, "superresolution", "model_srresnet.pth"),
        dataset="Set5",
        scale=s,
        gpus=0,
    )

    w, h = input_image.shape[0:2]
    cuda = opt.cuda

    if cuda:
        model = torch.load(opt.model)["model"]
    else:
        model = torch.load(opt.model, map_location=torch.device("cpu"))["model"]

    im_input = input_image.astype(np.float32).transpose(2, 0, 1)
    im_input = im_input.reshape(
        1, im_input.shape[0], im_input.shape[1], im_input.shape[2]
    )
    im_input = Variable(torch.from_numpy(im_input / 255.0).float())

    if cuda and not cpu_flag:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    if fFlag:
        im_h = np.zeros([4 * w, 4 * h, 3])
        wbin = 300
        i = 0
        while i < w:
            i_end = min(i + wbin, w)
            j = 0
            while j < h:
                j_end = min(j + wbin, h)
                patch = im_input[:, :, i:i_end, j:j_end]
                with torch.no_grad():
                    HR_4x = model(patch)
                HR_4x = HR_4x.cpu().data[0].numpy().astype(np.float32) * 255.0
                HR_4x = np.clip(HR_4x, 0.0, 255.0).transpose(1, 2, 0).astype(np.uint8)

                im_h[4 * i: 4 * i_end, 4 * j: 4 * j_end, :] = HR_4x
                j = j_end

            i = i_end
    else:
        with torch.no_grad():
            HR_4x = model(im_input)
        HR_4x = HR_4x.cpu()
        im_h = HR_4x.data[0].numpy().astype(np.float32)
        im_h = im_h * 255.0
        im_h = np.clip(im_h, 0.0, 255.0)
        im_h = im_h.transpose(1, 2, 0).astype(np.uint8)
    im_h = cv2.resize(im_h, (0, 0), fx=s / 4, fy=s / 4)
    return im_h


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    s = data_output["scale"]
    filter = data_output["filter"]
    image = cv2.imread(os.path.join(weight_path, "..", "cache.png"))[:, :, ::-1]
    try:
        output = get_super(
            image, s=s, cpu_flag=force_cpu, fFlag=filter, weight_path=weight_path
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