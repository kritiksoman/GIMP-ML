"""
Script will download weights and create gimp_ml_config.pkl, and print path to be added to GIMP Preferences.
"""
import os
import sys
import pickle
import csv
import hashlib
import gdown
import gimpml


def setup_python_weights(install_location=None):
    if not install_location:
        install_location = os.path.join(os.path.expanduser("~"), "GIMP-ML")
        # install_location = os.path.join(os.environ.get("HOMEDRIVE"), os.environ.get("HOMEPATH"), "GIMP-ML")
    if not os.path.isdir(install_location):
        os.mkdir(install_location)
    python_string = "python"
    if os.name == "nt":  # windows
        python_string += ".exe"
    python_path = os.path.join(os.path.dirname(sys.executable), python_string)
    weight_path = os.path.join(install_location, "weights")
    if not os.path.isdir(weight_path):
        os.mkdir(weight_path)

    step = 1
    print("\n##########")
    if os.name == "nt":  # windows
        print(
            "{}>> Automatic downloading of weights not supported on Windows.".format(
                step
            )
        )
        step += 1
        print(
            "{}>> Please downloads weights folder from: \n"
            "https://drive.google.com/drive/folders/1AtuIkGH7gqD9e5Tb-Y7wM9sLAZPyP_Mq?usp=sharing".format(
                step
            )
        )
        print("and place in: " + weight_path)
        step += 1
    else:  # linux
        file_path = os.path.join(os.path.dirname(gimpml.__file__), "tools")
        with open(os.path.join(file_path, "model_info.csv")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            headings = next(csv_reader)
            for row in csv_reader:
                model_path, file_id = os.path.join(*row[0].split("/")), row[1]
                file_size, model_file_name, md5sum = float(row[2]), row[3], row[4]
                if not os.path.isdir(os.path.join(weight_path, model_path)):
                    os.makedirs(os.path.join(weight_path, model_path))
                destination = os.path.join(
                    os.path.join(weight_path, model_path), model_file_name
                )
                if os.path.isfile(destination):
                    md5_hash = hashlib.md5()
                    a_file = open(destination, "rb")
                    content = a_file.read()
                    md5_hash.update(content)
                    digest = md5_hash.hexdigest()
                    a_file.close()
                if not os.path.isfile(destination) or (digest and digest != md5sum):
                    print(
                        "\nDownloading "
                        + model_path
                        + "(~"
                        + str(file_size)
                        + "MB)..."
                    )
                    url = "https://drive.google.com/uc?id={0}".format(file_id)
                    try:
                        gdown.cached_download(url, destination, md5=md5sum)
                    except:
                        try:
                            gdown.download(url, destination, quiet=False)
                        except:
                            print("Failed to download !")
    # plugin_loc = os.path.dirname(os.path.realpath(__file__))
    plugin_loc = os.path.dirname(gimpml.__file__)
    with open(os.path.join(plugin_loc, "tools", "gimp_ml_config.pkl"), "wb") as file:
        pickle.dump({"python_path": python_path, "weight_path": weight_path}, file)

    print(
        "{}>> Please add this path to Preferences --> Plug-ins in GIMP : ".format(step),
        os.path.join(plugin_loc, "plugins"),
    )
    print("##########\n")


if __name__ == "__main__":
    setup_python_weights()
