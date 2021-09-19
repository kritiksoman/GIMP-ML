import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "PyTorch-Image-Dehazing")
sys.path.extend([plugin_loc])

import torch
import net
import numpy as np
import cv2
from gimpml.tools.tools_utils import get_weight_path
import traceback


def get_dehaze(data_hazy, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    data_hazy = data_hazy / 255.0
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    dehaze_net = net.dehaze_net()

    if torch.cuda.is_available() and not cpu_flag:
        dehaze_net = dehaze_net.cuda()
        dehaze_net.load_state_dict(
            torch.load(os.path.join(weight_path, "deepdehaze", "dehazer.pth"))
        )
        data_hazy = data_hazy.cuda()
    else:
        dehaze_net.load_state_dict(
            torch.load(
                os.path.join(weight_path, "deepdehaze", "dehazer.pth"),
                map_location=torch.device("cpu"),
            )
        )

    data_hazy = data_hazy.unsqueeze(0)
    with torch.no_grad():
        clean_image = dehaze_net(data_hazy)
    out = clean_image.detach().cpu().numpy()[0, :, :, :] * 255
    out = np.clip(np.transpose(out, (1, 2, 0)), 0, 255).astype(np.uint8)
    return out


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    image = cv2.imread(os.path.join(weight_path, "..", "cache.png"))[:, :, ::-1]
    try:
        output = get_dehaze(image, cpu_flag=force_cpu, weight_path=weight_path)
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
