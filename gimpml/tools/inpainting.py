import pickle
import os
import sys

plugin_loc = os.path.dirname(os.path.realpath(__file__)) + '/'
sys.path.extend([plugin_loc + 'Inpainting'])

import torch
import numpy as np
from torch import nn
import scipy.ndimage
import cv2
from DFNet_core import DFNet
from RefinementNet_core import RefinementNet


def to_numpy(tensor):
    tensor = tensor.mul(255).byte().data.cpu().numpy()
    tensor = np.transpose(tensor, [0, 2, 3, 1])
    return tensor


def padding(img, height=512, width=512, channels=3):
    channels = img.shape[2] if len(img.shape) > 2 else 1
    interpolation = cv2.INTER_NEAREST

    if channels == 1:
        img_padded = np.zeros((height, width), dtype=img.dtype)
    else:
        img_padded = np.zeros((height, width, channels), dtype=img.dtype)

    original_shape = img.shape
    rows_rate = original_shape[0] / height
    cols_rate = original_shape[1] / width
    new_cols = width
    new_rows = height
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * height) // original_shape[0]
        img = cv2.resize(img, (new_cols, height), interpolation=interpolation)
        if new_cols > width:
            new_cols = width
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * width) // original_shape[1]
        img = cv2.resize(img, (width, new_rows), interpolation=interpolation)
        if new_rows > height:
            new_rows = height
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img
    return img_padded, new_cols, new_rows


def preprocess_image_dfnet(image, mask, model, device):
    image, new_cols, new_rows = padding(image, 512, 512)
    mask, _, _ = padding(mask, 512, 512)
    image = np.ascontiguousarray(image.transpose(2, 0, 1)).astype(np.uint8)
    mask = np.ascontiguousarray(np.expand_dims(mask, 0)).astype(np.uint8)

    image = torch.from_numpy(image).to(device).float().div(255)
    mask = 1 - torch.from_numpy(mask).to(device).float().div(255)
    image_miss = image * mask
    DFNET_output = model(image_miss.unsqueeze(0), mask.unsqueeze(0))[0]
    DFNET_output = image * mask + DFNET_output * (1 - mask)
    DFNET_output = to_numpy(DFNET_output)[0]
    DFNET_output = cv2.cvtColor(DFNET_output, cv2.COLOR_BGR2RGB)
    DFNET_output = DFNET_output[
                   (DFNET_output.shape[0] - new_rows) // 2: (DFNET_output.shape[0] - new_rows) // 2 + new_rows,
                   (DFNET_output.shape[1] - new_cols) // 2: (DFNET_output.shape[1] - new_cols) // 2 + new_cols, ...]

    return DFNET_output


def preprocess_image(image, mask, image_before_resize, model, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    shift_val = (100 / 512) * image.shape[0]

    image_resized = cv2.resize(image_before_resize, (image.shape[1], image.shape[0]))

    mask = mask // 255
    image_matched = image * (1 - mask) + image_resized * mask
    mask = mask * 255

    img_1 = scipy.ndimage.shift(image_matched, (-shift_val, 0, 0), order=0, mode='constant', cval=1)
    mask_1 = scipy.ndimage.shift(mask, (-shift_val, 0, 0), order=0, mode='constant', cval=255)
    img_2 = scipy.ndimage.shift(image_matched, (shift_val, 0, 0), order=0, mode='constant', cval=1)
    mask_2 = scipy.ndimage.shift(mask, (shift_val, 0, 0), order=0, mode='constant', cval=255)
    img_3 = scipy.ndimage.shift(image_matched, (0, shift_val, 0), order=0, mode='constant', cval=1)
    mask_3 = scipy.ndimage.shift(mask, (0, shift_val, 0), order=0, mode='constant', cval=255)
    img_4 = scipy.ndimage.shift(image_matched, (0, -shift_val, 0), order=0, mode='constant', cval=1)
    mask_4 = scipy.ndimage.shift(mask, (0, -shift_val, 0), order=0, mode='constant', cval=255)
    image_cat = np.dstack((mask, image_matched, img_1, mask_1, img_2, mask_2, img_3, mask_3, img_4, mask_4))

    mask_patch = torch.from_numpy(image_cat).to(device).float().div(255).unsqueeze(0)
    mask_patch = mask_patch.permute(0, -1, 1, 2)
    inputs = mask_patch[:, 1:, ...]
    mask = mask_patch[:, 0:1, ...]
    out = model(inputs, mask)
    out = out.mul(255).byte().data.cpu().numpy()
    out = np.transpose(out, [0, 2, 3, 1])[0]

    return out


def pad_image(image):
    x = ((image.shape[0] // 256) + (1 if image.shape[0] % 256 != 0 else 0)) * 256
    y = ((image.shape[1] // 256) + (1 if image.shape[1] % 256 != 0 else 0)) * 256
    padded = np.zeros((x, y, image.shape[2]), dtype='uint8')
    padded[:image.shape[0], :image.shape[1], ...] = image
    return padded


def get_inpaint(img, mask, cpu_flag=False):
    if torch.cuda.is_available() and not cpu_flag:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    mask = mask[..., :1]

    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    shape = image.shape

    image = pad_image(image)
    mask = pad_image(mask)

    DFNet_model = DFNet().to(device)
    weight_file_path = os.path.join(weight_path, "inpainting", "model_places2.pth")
    with open(r"C:\Users\Kritik Soman\GIMP-ML\output.txt", "a") as file:
        file.write(str(weight_file_path))
    DFNet_model.load_state_dict(torch.load(weight_file_path, map_location=device))
    DFNet_model.eval()
    DFNET_output = preprocess_image_dfnet(image, mask, DFNet_model, device)
    del DFNet_model
    Refinement_model = RefinementNet().to(device)
    weight_file_path = os.path.join(weight_path, "inpainting", "refinement.pth")
    Refinement_model.load_state_dict(
        torch.load(weight_file_path, map_location=device)['state_dict'])
    Refinement_model.eval()
    out = preprocess_image(image, mask, DFNET_output, Refinement_model, device)
    out = out[:shape[0], :shape[1], ...]
    del Refinement_model
    return out


if __name__ == "__main__":
    config_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(config_path, 'gimp_ml_config.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    weight_path = data_output["weight_path"]
    image1 = cv2.imread(os.path.join(weight_path, '..', "cache0.png"))[:, :, ::-1]
    image2 = cv2.imread(os.path.join(weight_path, '..', "cache1.png"))[:, :, ::-1]
    with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    if (np.sum(image1 == [0, 0, 0]) + np.sum(image1 == [255, 255, 255])) / (
            image1.shape[0] * image1.shape[1] * 3) > 0.8:
        output = get_inpaint(image2, image1, cpu_flag=force_cpu)
    else:
        output = get_inpaint(image1, image2, cpu_flag=force_cpu)
    cv2.imwrite(os.path.join(weight_path, '..', 'cache.png'), output[:, :, ::-1])
    # with open(os.path.join(weight_path, 'gimp_ml_run.pkl'), 'wb') as file:
    #     pickle.dump({"run_success": True}, file)
