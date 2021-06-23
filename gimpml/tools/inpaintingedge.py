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


def get_inpaint_edge(img, mask, cpu_flag=False, weight_path=None):
    config = Config('./config.yml')
    config._dict = {'MODE': 2, 'MODEL': 3, 'MASK': 3, 'EDGE': 1, 'NMS': 1, 'SEED': 10, 'GPU': [0], 'DEBUG': 0, 'VERBOSE': 0,
                    'TRAIN_FLIST': './datasets/places2_train.flist', 'VAL_FLIST': './datasets/places2_val.flist',
                    'TEST_FLIST': './examples/places2/images',
                    'TRAIN_EDGE_FLIST': './datasets/places2_edges_train.flist',
                    'VAL_EDGE_FLIST': './datasets/places2_edges_val.flist',
                    'TEST_EDGE_FLIST': './datasets/places2_edges_test.flist',
                    'TRAIN_MASK_FLIST': './datasets/masks_train.flist', 'VAL_MASK_FLIST': './datasets/masks_val.flist',
                    'TEST_MASK_FLIST': './examples/places2/masks', 'LR': 0.0001, 'D2G_LR': 0.1, 'BETA1': 0.0,
                    'BETA2': 0.9, 'BATCH_SIZE': 8, 'INPUT_SIZE': 256, 'SIGMA': 2, 'MAX_ITERS': '2e6', 'EDGE_THRESHOLD': 0.5,
                    'L1_LOSS_WEIGHT': 1, 'FM_LOSS_WEIGHT': 10, 'STYLE_LOSS_WEIGHT': 250, 'CONTENT_LOSS_WEIGHT': 0.1,
                    'INPAINT_ADV_LOSS_WEIGHT': 0.1, 'GAN_LOSS': 'nsgan', 'GAN_POOL_SIZE': 0, 'SAVE_INTERVAL': 1000,
                    'SAMPLE_INTERVAL': 1000, 'SAMPLE_SIZE': 12, 'EVAL_INTERVAL': 0, 'LOG_INTERVAL': 10,
                    'PATH': r'C:\Users\Kritik Soman\GIMP-ML\weights\edgeconnect\places2', 'RESULTS': r'C:\Users\Kritik Soman\GIMP-ML'}

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available():
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
    model.test()

if __name__ == "__main__":
    weight_path = get_weight_path()
    image1 = cv2.imread(os.path.join(weight_path, '..', "cache0.png"))[:, :, ::-1]
    image2 = cv2.imread(os.path.join(weight_path, '..', "cache1.png"))[:, :, ::-1]
    with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    if (np.sum(image1 == [0, 0, 0]) + np.sum(image1 == [255, 255, 255])) / (
            image1.shape[0] * image1.shape[1] * 3) > 0.8:
        output = get_inpaint_edge(image2, image1, cpu_flag=force_cpu, weight_path=weight_path)
    else:
        output = get_inpaint_edge(image1, image2, cpu_flag=force_cpu, weight_path=weight_path)
    cv2.imwrite(os.path.join(weight_path, '..', 'cache.png'), output[:, :, ::-1])
    # with open(os.path.join(weight_path, 'gimp_ml_run.pkl'), 'wb') as file:
    #     pickle.dump({"run_success": True}, file)
