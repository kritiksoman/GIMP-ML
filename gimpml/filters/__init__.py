import os
import pickle
from gimpml.tools.tools_utils import get_weight_path
import subprocess
import numpy as np
import cv2


# procedure="fog"
# image_path = r"D:\PycharmProjects\test_aug\dp.png"
# save_path = r"D:\PycharmProjects\test_aug\dp2.png"
# opacity = 100
# turbulence = 1  # between o and 1
# rgb = (244, 0, 0)
# image = cv2.imread(image_path)[:, :, ::-1]
# kwargs = {"image": image, "opacity": opacity, "rgb": rgb, "turbulence": turbulence}

def run(procedure, **kwargs):
    weight_path = get_weight_path()
    if "image_path" in kwargs.keys() and "image" in kwargs.keys():
        raise Exception("Both image_path and image should not be passed as input.")
    if "image_path" in kwargs.keys() and not os.path.isfile(kwargs['image_path']):
        raise Exception("Input image file does not exist.")
    if "image" in kwargs.keys() and not isinstance(kwargs['image'], np.ndarray) and not len(kwargs['image'].shape) == 3:
        raise Exception("Invalid input image.")
    return_image, remove_input_image = False, False
    if "save_path" not in kwargs.keys():
        kwargs["save_path"] = os.path.join(weight_path, "..", "tmp_filter2.png")
        return_image = True
    if "image" in kwargs.keys():
        image_path = os.path.join(weight_path, "..", "tmp_filter.png")
        channels = kwargs["image"].shape[2]
        if channels == 3:
            cv2.imwrite(image_path, kwargs["image"][:, :, ::-1])
            kwargs["image_path"] = image_path
            kwargs.pop("image")
            remove_input_image = True
            # print("Image saved.")
        elif channels == 4:
            cv2.imwrite(image_path, kwargs["image"][:, :, [2, 1, 0, 3]])
            kwargs["image_path"] = image_path
            kwargs.pop("image")
            remove_input_image = True
            # print("Image saved.")
        else:
            raise Exception("High-dimensional image not supported.")
    with open(os.path.join(weight_path, "..", "gimp_ml_augment.pkl"), "wb") as file:
        pickle.dump(
            kwargs,
            file,
        )
    command_str = "gimp-2.99 -idf --batch-interpreter=python-fu-eval -b - < " + os.path.join(
        os.path.dirname(os.path.realpath(__file__)), procedure.strip()
    ) + ".py"
    # print(command_str)
    subprocess.call(command_str, shell=True)
    if remove_input_image:
        os.remove(kwargs["image_path"])
    if return_image:
        img = cv2.imread(kwargs["save_path"])
        channels = img.shape[2]
        os.remove(kwargs["save_path"])
        return img[:, :, ::-1] if channels == 3 else img[:, :, [2, 1, 0, 3]]

