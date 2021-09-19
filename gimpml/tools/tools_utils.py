import os
import pickle


def get_weight_path():
    config_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(config_path, "gimp_ml_config.pkl"), "rb") as file:
        data_output = pickle.load(file)
    weight_path = data_output["weight_path"]
    return weight_path
