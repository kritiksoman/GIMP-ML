import cv2
import pickle
import subprocess


def run_plugin(data_input, plugin_path):
    dbfile = open(data_path + "data_input", 'ab')
    pickle.dump(data_input, dbfile)  # source, destination
    dbfile.close()
    subprocess.call([python_path, plugin_path])
    dbfile = open(data_path + "data_output", 'rb')
    data_output = pickle.load(dbfile)
    dbfile.close()
    return data_output


image = cv2.imread('/Users/kritiksoman/Documents/GitHub/GIMP-ML-pip/sampleinput/inpaint.png')[:, :, ::-1]

python_path = "/Users/kritiksoman/GIMP-ML/gimpenv3/bin/python"
plugin_path = "/Users/kritiksoman/Documents/GitHub/GIMP3-ML-pip/gimpml/monodepth.py"
data_path = "/Users/kritiksoman/GIMP-ML/"

# save data and call plugin
data_input = {'image': image, 'args_input': {'force_cpu': 0}}

data_output = run_plugin(data_input, plugin_path)
# print(data_output)
cv2.imwrite("output.png", data_output['image_output'])

