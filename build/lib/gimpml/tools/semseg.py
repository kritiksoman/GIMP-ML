import pickle
import os
import sys

# plugin_loc = os.path.dirname(os.path.realpath(__file__)) + '/'
# sys.path.extend([plugin_loc + 'MiDaS'])

from PIL import Image
import torch
from torchvision import transforms, datasets
import numpy as np
import cv2
import os


def get_seg(input_image, cpu_flag=False):
    model = torch.load(os.path.join(weight_path, 'deeplabv3', 'deeplabv3+model.pt'))
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_image = Image.fromarray(input_image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    if torch.cuda.is_available() and not cpu_flag:
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)

    tmp = np.array(r)
    tmp2 = 10 * np.repeat(tmp[:, :, np.newaxis], 3, axis=2)
    return tmp2


if __name__ == "__main__":
    config_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(config_path, 'gimp_ml_config.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    weight_path = data_output["weight_path"]
    image = cv2.imread(os.path.join(weight_path, '..', "cache.png"))[:, :, ::-1]
    with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    output = get_seg(image, cpu_flag=force_cpu)
    cv2.imwrite(os.path.join(weight_path, '..', 'cache.png'), output[:, :, ::-1])
    # with open(os.path.join(weight_path, 'gimp_ml_run.pkl'), 'wb') as file:
    #     pickle.dump({"run_success": True}, file)
