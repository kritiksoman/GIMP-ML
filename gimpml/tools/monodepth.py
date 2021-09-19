import pickle
import os
import sys
import traceback

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "DPT")
sys.path.extend([plugin_loc])

from monodepth_run import run
import numpy as np
import cv2
import torch
from gimpml.tools.tools_utils import get_weight_path


def get_mono_depth(input_image, cpu_flag=False, weight_path=None, absolute_depth=False):
    if weight_path is None:
        weight_path = get_weight_path()

    with torch.no_grad():
        out = run(
            input_image,
            os.path.join(weight_path, "MiDaS", "dpt_hybrid-midas-501f0c75.pt"),
            cpu_flag=cpu_flag,
            bits=2,
            absolute_depth=absolute_depth,
        )

    out = np.repeat(out[:, :, np.newaxis], 3, axis=2)
    d1, d2 = input_image.shape[:2]
    out = cv2.resize(out, (d2, d1))
    return out


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    image = cv2.imread(os.path.join(weight_path, "..", "cache.png"))[:, :, ::-1]
    force_cpu = data_output["force_cpu"]
    try:
        output = get_mono_depth(image, cpu_flag=force_cpu, weight_path=weight_path)

        cv2.imwrite(
            os.path.join(weight_path, "..", "cache.png"),
            output.astype("uint16"),
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
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
