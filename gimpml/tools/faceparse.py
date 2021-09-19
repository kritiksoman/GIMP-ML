import pickle
import os
import sys

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "face-parsing-PyTorch")
sys.path.extend([plugin_loc])

from model import BiSeNet
from PIL import Image
import torch
from torchvision import transforms, datasets
import numpy as np
import cv2
import os
from gimpml.tools.tools_utils import get_weight_path
import traceback


colors = np.array(
    [
        [0, 0, 0],
        [204, 0, 0],
        [0, 255, 255],
        [51, 255, 255],
        [51, 51, 255],
        [204, 0, 204],
        [204, 204, 0],
        [102, 51, 0],
        [255, 0, 0],
        [0, 204, 204],
        [76, 153, 0],
        [102, 204, 0],
        [255, 255, 0],
        [0, 0, 153],
        [255, 153, 51],
        [0, 51, 0],
        [0, 204, 0],
        [0, 0, 204],
        [255, 51, 153],
    ]
)
colors = colors.astype(np.uint8)


def getlabelmat(mask, idx):
    x = np.zeros((mask.shape[0], mask.shape[1], 3))
    x[mask == idx, 0] = colors[idx][0]
    x[mask == idx, 1] = colors[idx][1]
    x[mask == idx, 2] = colors[idx][2]
    return x


def colorMask(mask):
    x = np.zeros((mask.shape[0], mask.shape[1], 3))
    for idx in range(19):
        x = x + getlabelmat(mask, idx)
    return np.uint8(x)


def get_face(input_image, cpu_flag=False, weight_path=None):
    if weight_path is None:
        weight_path = get_weight_path()
    save_pth = os.path.join(weight_path, "faceparse", "79999_iter.pth")
    input_image = Image.fromarray(input_image)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    if torch.cuda.is_available() and not cpu_flag:
        net.cuda()
        net.load_state_dict(torch.load(save_pth))
    else:
        net.load_state_dict(
            torch.load(save_pth, map_location=lambda storage, loc: storage)
        )

    net.eval()

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    with torch.no_grad():
        img = input_image.resize((512, 512), Image.BILINEAR)
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        if torch.cuda.is_available() and not cpu_flag:
            img = img.cuda()
        out = net(img)[0]
        if torch.cuda.is_available():
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        else:
            parsing = out.squeeze(0).numpy().argmax(0)

    parsing = Image.fromarray(np.uint8(parsing))
    parsing = parsing.resize(input_image.size)
    parsing = np.array(parsing)
    parsing = colorMask(parsing)
    return parsing


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    force_cpu = data_output["force_cpu"]
    image = cv2.imread(os.path.join(weight_path, "..", "cache.png"))[:, :, ::-1]
    try:
        output = get_face(image, cpu_flag=force_cpu, weight_path=weight_path)
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
