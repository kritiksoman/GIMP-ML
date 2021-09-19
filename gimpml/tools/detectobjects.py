import sys
import os

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pytorch-YOLOv4")
sys.path.extend([plugin_loc])

import cv2
import torch
from yolo_models import Yolov4
from yolo_tool.utils import load_class_names, plot_boxes_cv2, get_objects
from yolo_tool.torch_utils import do_detect
from gimpml.tools.tools_utils import get_weight_path
import pickle
import traceback
import shutil
import numpy as np


def scale_image(image):
    height, width = image.shape[:2]
    m, n = 0, 0
    if height > 320 or width > 320:
        if height > 320:
            n = (height - 320) / 96
        if width > 320:
            m = (width - 320) / 96
        sized = cv2.resize(image, (320 + 96 * int(np.ceil(m)), 320 + 96 * int(np.ceil(n))))
    else:
        sized = cv2.resize(image, (320, 320))
    scale_width = sized.shape[1] / width
    scale_height = sized.shape[0] / height
    return sized, scale_width, scale_height


def get_detect_objects(image=None, image_path=None, cpu_flag=False, weight_path=None, get_predict_image=False,
                       n_classes=80):
    if image is not None and image_path is not None:
        raise Exception("Invalid input.")

    if weight_path is None:
        weight_path = get_weight_path()

    weightfile = os.path.join(weight_path, "yolo", "yolov4.pth")

    use_cuda = False
    if torch.cuda.is_available() and not cpu_flag:
        use_cuda = True

    if n_classes == 20:
        namesfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pytorch-YOLOv4", "yolo_data", "voc.names")
    elif n_classes == 80:
        namesfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pytorch-YOLOv4", "yolo_data", "coco.names")
    else:
        print("please give namefile")
    class_names = load_class_names(namesfile)

    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
    if use_cuda:
        pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    else:
        pretrained_dict = torch.load(weightfile, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_dict)
    if use_cuda:
        model.cuda()

    result = []
    if image is not None and image_path is None:
        sized, scale_width, scale_height = scale_image(image)
        with torch.no_grad():
            boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
        result.append(plot_boxes_cv2(image, boxes[0], scale_width, scale_height, class_names=class_names) if get_predict_image else
                      (get_objects(image, boxes[0], scale_width, scale_height, class_names), "image"))
    elif image is None and image_path is not None:
        for filename in os.listdir(image_path):
            try:
                image = cv2.imread(os.path.join(image_path, filename))[:, :, [2, 1, 0]]
                sized, scale_width, scale_height = scale_image(image)
                with torch.no_grad():
                    boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
                result.append(plot_boxes_cv2(sized, boxes[0], scale_width, scale_height, class_names=class_names) if get_predict_image else
                              (get_objects(sized, boxes[0], scale_width, scale_height, class_names),
                               os.path.join(image_path, filename)))
            except:
                pass
    return result


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    if "force_cpu" in data_output.keys():
        force_cpu = data_output["force_cpu"]
    else:
        force_cpu = False if torch.cuda.is_available() else True
    get_predict_image = data_output["get_predict_image"]
    image1, image_path = None, None
    if get_predict_image:
        image1 = cv2.imread(os.path.join(weight_path, "..", "cache.png"))
    else:
        image_path = data_output["image_path"]
        search_objects = [x.lower().strip() for x in data_output["objects"].split("|")]
    try:
        if get_predict_image:
            count = 0
            output = get_detect_objects(image=image1, cpu_flag=force_cpu, weight_path=weight_path, get_predict_image=True)[0]
            cv2.imwrite(os.path.join(weight_path, "..", "cache.png"), output)
        else:
            count = 0
            output = get_detect_objects(image_path=image_path, cpu_flag=force_cpu, weight_path=weight_path)
            save_filtered_path = os.path.join(image_path, "filtered")
            if not os.path.exists(save_filtered_path):
                os.makedirs(save_filtered_path)
            for res in output:
                if any([obj[-2] in search_objects for obj in res[0]]):
                    head, tail = os.path.split(res[-1])
                    shutil.move(res[-1], os.path.join(head, "filtered", tail))
                    count += 1

        with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "wb") as file:
            pickle.dump({"inference_status": "success", "force_cpu": force_cpu, "count": count}, file)

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
