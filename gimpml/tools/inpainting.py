import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "edge-connect")
sys.path.extend([plugin_loc])

import torch
import cv2
import numpy as np
from src.edge_connect import EdgeConnect
import random
import os
from src.config import Config
from skimage.feature import canny
from gimpml.tools.tools_utils import get_weight_path
import traceback


def get_inpaint(images, masks, cpu_flag=False, model_name="places2", weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    config = Config()
    config._dict = {
        "MODE": 2,
        "MODEL": 3,
        "MASK": 3,
        "EDGE": 1,
        "NMS": 1,
        "SEED": 10,
        "GPU": [0],
        "DEBUG": 0,
        "VERBOSE": 0,
        "LR": 0.0001,
        "D2G_LR": 0.1,
        "BETA1": 0.0,
        "BETA2": 0.9,
        "BATCH_SIZE": 8,
        "INPUT_SIZE": 256,
        "SIGMA": 2,
        "MAX_ITERS": "2e6",
        "EDGE_THRESHOLD": 0.5,
        "L1_LOSS_WEIGHT": 1,
        "FM_LOSS_WEIGHT": 10,
        "STYLE_LOSS_WEIGHT": 250,
        "CONTENT_LOSS_WEIGHT": 0.1,
        "INPAINT_ADV_LOSS_WEIGHT": 0.1,
        "GAN_LOSS": "nsgan",
        "GAN_POOL_SIZE": 0,
        "SAVE_INTERVAL": 1000,
        "SAMPLE_INTERVAL": 1000,
        "SAMPLE_SIZE": 12,
        "EVAL_INTERVAL": 0,
        "LOG_INTERVAL": 10,
        "PATH": os.path.join(weight_path, "edgeconnect", model_name),
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available() and not cpu_flag:
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = EdgeConnect(config)
    model.load()
    images_gray = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)
    sigma = config.SIGMA

    if sigma == -1:
        sigma = random.randint(1, 4)

    masks = masks / 255
    edge = canny(images_gray, sigma=sigma, mask=(1 - masks).astype(bool)).astype(
        np.float32
    )
    images_gray = images_gray / 255
    images = images / 255

    images = (
        torch.from_numpy(images.astype(np.float32).copy())
        .permute((2, 0, 1))
        .unsqueeze(0)
    )
    images_gray = (
        torch.from_numpy(images_gray.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )
    masks = torch.from_numpy(masks.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    edges = torch.from_numpy(edge.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    model.edge_model.eval()
    model.inpaint_model.eval()

    if config.DEVICE.type == "cuda":
        images, images_gray, edges, masks = model.cuda(
            *(images, images_gray, edges, masks)
        )

    # edge model
    if config.MODEL == 1:
        with torch.no_grad():
            outputs = model.edge_model(images_gray, edges, masks)
        outputs_merged = (outputs * masks) + (edges * (1 - masks))

    # inpaint model
    elif config.MODEL == 2:
        with torch.no_grad():
            outputs = model.inpaint_model(images, edges, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

    # inpaint with edge model / joint model
    else:
        with torch.no_grad():
            edges = model.edge_model(images_gray, edges, masks).detach()
            outputs = model.inpaint_model(images, edges, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

    output = model.postprocess(outputs_merged)[0]
    return np.uint8(output.cpu())


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    n_drawables = data_output["n_drawables"]
    image1 = cv2.imread(os.path.join(weight_path, "..", "cache0.png"))
    image2 = None
    if n_drawables == 2:
        image2 = cv2.imread(os.path.join(weight_path, "..", "cache1.png"))
    force_cpu = data_output["force_cpu"]
    model_name = data_output["model_name"]
    h, w, c = image1.shape
    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))

    try:
        if (np.sum(image1 == [0, 0, 0]) + np.sum(image1 == [255, 255, 255])) / (
            image1.shape[0] * image1.shape[1] * 3
        ) > 0.8:
            output = get_inpaint(
                image2[:, :, ::-1],
                image1[:, :, 0],
                cpu_flag=force_cpu,
                model_name=model_name,
                weight_path=weight_path,
            )
        else:
            output = get_inpaint(
                image1[:, :, ::-1],
                image2[:, :, 0],
                cpu_flag=force_cpu,
                model_name=model_name,
                weight_path=weight_path,
            )
        output = cv2.resize(output, (w, h))
        cv2.imwrite(os.path.join(weight_path, "..", "cache.png"), output[:, :, ::-1])
        with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "wb") as file:
            pickle.dump(
                {
                    "inference_status": "success",
                    "force_cpu": force_cpu,
                    "model_name": model_name,
                    "n_drawables": n_drawables,
                },
                file,
            )

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
