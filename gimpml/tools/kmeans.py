import pickle
import os
import sys


import numpy as np
from scipy.cluster.vq import kmeans2
import cv2

def get_kmeans(image, locflag=False, n_clusters=3):
    if image.shape[2] == 4:  # get rid of alpha channel
        image = image[:, :, 0:3]
    h, w, d = image.shape
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))

    if locflag:
        xx, yy = np.meshgrid(range(w), range(h))
        x = xx.reshape(-1, 1)
        y = yy.reshape(-1, 1)
        pixel_values = np.concatenate((pixel_values, x, y), axis=1)

    pixel_values = np.float32(pixel_values)
    c, out = kmeans2(pixel_values, n_clusters)

    if locflag:
        c = np.uint8(c[:, 0:3])
    else:
        c = np.uint8(c)
    segmented_image = c[out.flatten()]
    segmented_image = segmented_image.reshape((h, w, d))
    return segmented_image


if __name__ == "__main__":
    config_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(config_path, 'gimp_ml_config.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    weight_path = data_output["weight_path"]
    image = cv2.imread(os.path.join(weight_path, '..', "cache.png"))[:, :, ::-1]
    with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    n_cluster = data_output["n_cluster"]
    position = data_output["position"]
    output = get_kmeans(image, locflag=position, n_clusters=n_cluster)
    cv2.imwrite(os.path.join(weight_path, '..', 'cache.png'), output[:, :, ::-1])
    # with open(os.path.join(weight_path, 'gimp_ml_run.pkl'), 'wb') as file:
    #     pickle.dump({"run_success": True}, file)
