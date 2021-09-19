import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "PD-Denoising-pytorch")
sys.path.extend([plugin_loc])

from denoiser import *
from argparse import Namespace
import torch
import cv2
import numpy as np
from gimpml.tools.tools_utils import get_weight_path
import traceback


def get_denoise(Img, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    w, h, _ = Img.shape
    opt = Namespace(
        color=1,
        cond=1,
        delog="logsdc",
        ext_test_noise_level=None,
        k=0,
        keep_ind=None,
        mode="MC",
        num_of_layers=20,
        out_dir="results_bc",
        output_map=0,
        ps=2,
        ps_scale=2,
        real_n=1,
        refine=0,
        refine_opt=1,
        rescale=1,
        scale=1,
        spat_n=0,
        test_data="real_night",
        test_data_gnd="Set12",
        test_noise_level=None,
        wbin=512,
        zeroout=0,
    )
    c = 1 if opt.color == 0 else 3
    model = DnCNN_c(channels=c, num_of_layers=opt.num_of_layers, num_of_est=2 * c)
    model_est = Estimation_direct(c, 2 * c)
    # device_ids = [0]
    # model = nn.DataParallel(net, device_ids=device_ids)
    # model_est = nn.DataParallel(est_net, device_ids=device_ids)# Estimator Model
    if torch.cuda.is_available() and not cpu_flag:
        ckpt_est = torch.load(os.path.join(weight_path, "deepdenoise", "est_net.pth"))
        ckpt = torch.load(os.path.join(weight_path, "deepdenoise", "net.pth"))
        model = model.cuda()
        model_est = model_est.cuda()
    else:
        ckpt = torch.load(
            os.path.join(weight_path, "deepdenoise", "net.pth"),
            map_location=torch.device("cpu"),
        )
        ckpt_est = torch.load(
            os.path.join(weight_path, "deepdenoise", "est_net.pth"),
            map_location=torch.device("cpu"),
        )

    ckpt = {key.replace("module.", ""): value for key, value in ckpt.items()}
    ckpt_est = {key.replace("module.", ""): value for key, value in ckpt_est.items()}

    model.load_state_dict(ckpt)
    model.eval()
    model_est.load_state_dict(ckpt_est)
    model_est.eval()
    try:
        gimp.progress_update(float(0.005))
        gimp.displays_flush()
    except:
        pass
    Img = Img[:, :, ::-1]  # change it to RGB
    Img = cv2.resize(Img, (0, 0), fx=opt.scale, fy=opt.scale)
    if opt.color == 0:
        Img = Img[:, :, 0]  # For gray images
        Img = np.expand_dims(Img, 2)
    pss = 1
    if opt.ps == 1:
        pss = decide_scale_factor(
            Img / 255.0,
            model_est,
            color=opt.color,
            thre=0.008,
            plot_flag=1,
            stopping=4,
            mark=opt.out_dir + "/" + file_name,
        )[0]
        # print(pss)
        Img = pixelshuffle(Img, pss)
    elif opt.ps == 2:
        pss = opt.ps_scale

    merge_out = np.zeros([w, h, 3])
    wbin = opt.wbin
    i = 0
    idx = 0
    t = float(w * h) / float(wbin * wbin)
    while i < w:
        i_end = min(i + wbin, w)
        j = 0
        while j < h:
            j_end = min(j + wbin, h)
            patch = Img[i:i_end, j:j_end, :]
            with torch.no_grad():
                patch_merge_out_numpy = denoiser(
                    patch, c, pss, model, model_est, opt, cpu_flag
                )
            merge_out[i:i_end, j:j_end, :] = patch_merge_out_numpy
            j = j_end
            idx = idx + 1
        i = i_end

    return merge_out[:, :, ::-1]


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    image = cv2.imread(os.path.join(weight_path, "..", "cache.png"))[:, :, ::-1]
    try:
        output = get_denoise(image, cpu_flag=force_cpu, weight_path=weight_path)
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
