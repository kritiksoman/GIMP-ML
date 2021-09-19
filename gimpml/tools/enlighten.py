import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "EnlightenGAN")
sys.path.extend([plugin_loc])

from argparse import Namespace
import cv2
import numpy as np
import torch
from enlighten_models.models import create_model
from enlighten_data.base_dataset import get_transform
from gimpml.tools.tools_utils import get_weight_path
import traceback


def get_enlighten(input_image, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    opt = Namespace(
        D_P_times2=False,
        IN_vgg=False,
        aspect_ratio=1.0,
        batchSize=1,
        checkpoints_dir=weight_path,
        dataroot="test_dataset",
        dataset_mode="unaligned",
        display_id=1,
        display_port=8097,
        display_single_pane_ncols=0,
        display_winsize=256,
        fcn=0,
        fineSize=256,
        gpu_ids=[0],
        high_times=400,
        how_many=50,
        hybrid_loss=False,
        identity=0.0,
        input_linear=False,
        input_nc=3,
        instance_norm=0.0,
        isTrain=False,
        l1=10.0,
        lambda_A=10.0,
        lambda_B=10.0,
        latent_norm=False,
        latent_threshold=False,
        lighten=False,
        linear=False,
        linear_add=False,
        loadSize=286,
        low_times=200,
        max_dataset_size="inf",
        model="single",
        multiply=False,
        nThreads=1,
        n_layers_D=3,
        n_layers_patchD=3,
        name="enlightening",
        ndf=64,
        new_lr=False,
        ngf=64,
        no_dropout=True,
        no_flip=True,
        no_vgg_instance=False,
        noise=0,
        norm="instance",
        norm_attention=False,
        ntest="inf",
        output_nc=3,
        patchD=False,
        patchD_3=0,
        patchSize=64,
        patch_vgg=False,
        phase="test",
        resize_or_crop="no",
        results_dir="./results/",
        self_attention=True,
        serial_batches=True,
        skip=1.0,
        syn_norm=False,
        tanh=False,
        times_residual=True,
        use_avgpool=0,
        use_mse=False,
        use_norm=1.0,
        use_ragan=False,
        use_wgan=0.0,
        vary=1,
        vgg=0,
        vgg_choose="relu5_3",
        vgg_maxpooling=False,
        vgg_mean=False,
        which_direction="AtoB",
        which_epoch="200",
        which_model_netD="basic",
        which_model_netG="sid_unet_resize",
        cFlag=cpu_flag,
    )

    im = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    transform = get_transform(opt)
    A_img = transform(im)
    r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
    A_gray = 1.0 - (0.299 * r + 0.587 * g + 0.114 * b) / 2.0
    A_gray = torch.unsqueeze(A_gray, 0)
    data = {
        "A": A_img.unsqueeze(0),
        "B": A_img.unsqueeze(0),
        "A_gray": A_gray.unsqueeze(0),
        "input_img": A_img.unsqueeze(0),
        "A_paths": "A_path",
        "B_paths": "B_path",
    }

    model = create_model(opt)
    model.set_input(data)
    with torch.no_grad():
        visuals = model.predict()
    out = visuals["fake_B"].astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    image = cv2.imread(os.path.join(weight_path, "..", "cache.png"))[:, :, ::-1]
    try:
        output = get_enlighten(image, cpu_flag=force_cpu, weight_path=weight_path)
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
