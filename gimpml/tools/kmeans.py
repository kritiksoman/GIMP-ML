import pickle
import os
import numpy as np
from scipy.cluster.vq import kmeans2
import cv2
from gimpml.tools.tools_utils import get_weight_path
import traceback
import sys


def get_kmeans(image, loc_flag=False, n_clusters=3):
    if image.shape[2] == 4:  # get rid of alpha channel
        image = image[:, :, 0:3]
    h, w, d = image.shape
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))

    if loc_flag:
        xx, yy = np.meshgrid(range(w), range(h))
        x = xx.reshape(-1, 1)
        y = yy.reshape(-1, 1)
        pixel_values = np.concatenate((pixel_values, x, y), axis=1)

    pixel_values = np.float32(pixel_values)
    c, out = kmeans2(pixel_values, n_clusters)

    if loc_flag:
        c = np.uint8(c[:, 0:3])
    else:
        c = np.uint8(c)
    segmented_image = c[out.flatten()]
    segmented_image = segmented_image.reshape((h, w, d))
    return segmented_image


if __name__ == "__main__":
    weight_path = get_weight_path()
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    n_cluster = data_output["n_cluster"]
    position = data_output["position"]
    image = cv2.imread(os.path.join(weight_path, "..", "cache.png"))[:, :, ::-1]
    try:
        output = get_kmeans(image, loc_flag=position, n_clusters=n_cluster)
        cv2.imwrite(os.path.join(weight_path, "..", "cache.png"), output[:, :, ::-1])
        with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "wb") as file:
            pickle.dump({"inference_status": "success"}, file)

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
