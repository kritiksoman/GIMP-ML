import pickle
import os
import sys

plugin_loc = os.path.dirname(os.path.realpath(__file__)) + '/'
sys.path.extend([plugin_loc + 'edge-connect'])

import torch
import cv2
import numpy as np
from src.edge_connect import EdgeConnect
import random
import os
from src.config import Config



def get_weight_path():
    config_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(config_path, 'gimp_ml_config.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    weight_path = data_output["weight_path"]
    return weight_path

def get_inpaint(images, masks, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    config = Config()
    config._dict = {'MODE': 2, 'MODEL': 3, 'MASK': 3, 'EDGE': 1, 'NMS': 1, 'SEED': 10, 'GPU': [0], 'DEBUG': 0,
                    'VERBOSE': 0,
                    'LR': 0.0001, 'D2G_LR': 0.1, 'BETA1': 0.0,
                    'BETA2': 0.9, 'BATCH_SIZE': 8, 'INPUT_SIZE': 256, 'SIGMA': 2, 'MAX_ITERS': '2e6',
                    'EDGE_THRESHOLD': 0.5,
                    'L1_LOSS_WEIGHT': 1, 'FM_LOSS_WEIGHT': 10, 'STYLE_LOSS_WEIGHT': 250, 'CONTENT_LOSS_WEIGHT': 0.1,
                    'INPAINT_ADV_LOSS_WEIGHT': 0.1, 'GAN_LOSS': 'nsgan', 'GAN_POOL_SIZE': 0, 'SAVE_INTERVAL': 1000,
                    'SAMPLE_INTERVAL': 1000, 'SAMPLE_SIZE': 12, 'EVAL_INTERVAL': 0, 'LOG_INTERVAL': 10,
                    'PATH': os.path.join(weight_path, 'edgeconnect', 'places2')}

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

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
    # model.test()

    images_gray = cv2.cvtColor(images, cv2.COLOR_RGB2GRAY)

    masks = masks/255
    sigma = config.SIGMA
    # TODO: fix canny edge
    if not sigma % 2:
        sigma += 1
    max_val = np.max(images_gray)
    img = cv2.GaussianBlur(images_gray, (sigma * 3, sigma * 3), sigma)
    img = cv2.Canny(img, 0.1 * max_val, 0.2 * max_val)
    edge = img * (1 - masks.astype(float))
    images_gray = images_gray / 255
    images = images / 255

    images = torch.from_numpy(images.astype(np.float32).copy()).permute((2, 0, 1)).unsqueeze(0)
    images_gray = torch.from_numpy(images_gray.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    masks = torch.from_numpy(masks.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    edges = torch.from_numpy(edge.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    model.edge_model.eval()
    model.inpaint_model.eval()

    if config.DEVICE.type == 'cuda':
        images, images_gray, edges, masks = model.cuda(*(images, images_gray, edges, masks))

    # edge model
    if config.MODEL == 1:
        outputs = model.edge_model(images_gray, edges, masks)
        outputs_merged = (outputs * masks) + (edges * (1 - masks))

    # inpaint model
    elif config.MODEL == 2:
        outputs = model.inpaint_model(images, edges, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

    # inpaint with edge model / joint model
    else:
        edges = model.edge_model(images_gray, edges, masks).detach()
        outputs = model.inpaint_model(images, edges, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

    output = model.postprocess(outputs_merged)[0]
    return np.uint8(output)


if __name__ == "__main__":
    weight_path = get_weight_path()
    image1 = cv2.imread(os.path.join(weight_path, '..', "cache0.png"))[:, :, ::-1]
    image2 = cv2.imread(os.path.join(weight_path, '..', "cache1.png"))[:, :, ::-1]
    with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    if (np.sum(image1 == [0, 0, 0]) + np.sum(image1 == [255, 255, 255])) / (
            image1.shape[0] * image1.shape[1] * 3) > 0.8:
        output = get_inpaint(image2, image1[:, :, 0], cpu_flag=force_cpu, weight_path=weight_path)
    else:
        output = get_inpaint(image1, image2[:, :, 0], cpu_flag=force_cpu, weight_path=weight_path)
    cv2.imwrite(os.path.join(weight_path, '..', 'cache.png'), output[:, :, ::-1])
    # with open(os.path.join(weight_path, 'gimp_ml_run.pkl'), 'wb') as file:
    #     pickle.dump({"run_success": True}, file)
