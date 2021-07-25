import pickle
import os
import sys

plugin_loc = os.path.dirname(os.path.realpath(__file__)) + '/'
sys.path.extend([plugin_loc + 'DPT'])

from semseg_run import run
from PIL import Image
import torch
from torchvision import transforms, datasets
import numpy as np
import cv2
import os
import traceback

def get_weight_path():
    config_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(config_path, 'gimp_ml_config.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    weight_path = data_output["weight_path"]
    return weight_path


def get_seg(input_image, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()

    with torch.no_grad():
        out = run(input_image, os.path.join(weight_path, 'semseg', 'dpt_hybrid-ade20k-53898607.pt'), cpu_flag=cpu_flag)
        # out = run_depth(image, os.path.join(weight_path, 'MiDaS', 'model.pt'), MonoDepthNet, MiDaS_utils, target_w=640,
        #                 f=cpu_flag)
    # out = np.repeat(out[:, :, np.newaxis], 3, axis=2)
    # d1, d2 = input_image.shape[:2]
    # out = cv2.resize(out, (d2, d1))
    return out


# def get_seg(input_image, cpu_flag=False, weight_path=None):
#     if weight_path is None:
#         weight_path = get_weight_path()
#     model = torch.load(os.path.join(weight_path, 'deeplabv3', 'deeplabv3+model.pt'))
#     model.eval()
#     preprocess = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     input_image = Image.fromarray(input_image)
#     input_tensor = preprocess(input_image)
#     input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
#     if torch.cuda.is_available() and not cpu_flag:
#         input_batch = input_batch.to('cuda')
#         model.to('cuda')
#
#     with torch.no_grad():
#         output = model(input_batch)['out'][0]
#     output_predictions = output.argmax(0)
#
#     # create a color pallette, selecting a color for each class
#     palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
#     colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
#     colors = (colors % 255).numpy().astype("uint8")
#
#     # plot the semantic segmentation predictions of 21 classes in each color
#     r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
#
#     tmp = np.array(r)
#     tmp2 = 10 * np.repeat(tmp[:, :, np.newaxis], 3, axis=2)
#     return tmp2


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    image = cv2.imread(os.path.join(weight_path, '..', "cache.png"))[:, :, ::-1]
    try:
        output = get_seg(image, cpu_flag=force_cpu, weight_path=weight_path)
        cv2.imwrite(os.path.join(weight_path, '..', 'cache.png'), output[:, :, ::-1])
        with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'wb') as file:
            pickle.dump({"inference_status": "success"}, file)

        # Remove old temporary error files that were saved
        my_dir = os.path.join(weight_path, '..')
        for f_name in os.listdir(my_dir):
            if f_name.startswith("error_log"):
                os.remove(os.path.join(my_dir, f_name))

    except Exception as error:
        with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'wb') as file:
            pickle.dump({"inference_status": "failed"}, file)
        with open(os.path.join(weight_path, '..', 'error_log.txt'), 'w') as f:
            f.write(str(error))
            f.write(traceback.format_exc())