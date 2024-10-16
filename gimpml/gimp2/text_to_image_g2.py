#!/usr/bin/env python2
# coding: utf-8
"""
 .d8888b.  8888888 888b     d888 8888888b.       888b     d888 888
d88P  Y88b   888   8888b   d8888 888   Y88b      8888b   d8888 888
888    888   888   88888b.d88888 888    888      88888b.d88888 888
888          888   888Y88888P888 888   d88P      888Y88888P888 888
888  88888   888   888 Y888P 888 8888888P"       888 Y888P 888 888
888    888   888   888  Y8P  888 888             888  Y8P  888 888
Y88b  d88P   888   888   "   888 888             888   "   888 888
 "Y8888P88 8888888 888       888 888             888       888 88888888
Convert Text to image.
"""

import threading
import sys, os
sys.path.extend([os.path.dirname(os.path.realpath(__file__))])
from plugin_utils_g2 import *
import time
# from gimpfu import *
import urllib2 
import base64
import json


v = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config.json")))
url = r"http://localhost:"+ str(v['gimpml']['port']) 
post_json = {
    "pipeline": "text_to_image",
    "model": "text_to_image",
    "text": "TEXT",
    "source": "gimp2"
}

gettext.install("gimp20-python", gimp.locale_directory, unicode=True)

models = ["standard", "hd"]

def text_to_image(img, layer, text, dropdown_index):
    # Mark undo
    gimp.context_push()
    img.undo_group_start()

    # Load/set model
    post_json["model"] = models[dropdown_index]
    post_json["pipeline"] = "text_to_image"
    post_request_string = json.dumps(post_json).encode('utf-8')
    req =  urllib2.Request(url + "/download_load_model", data=post_request_string) 
    output_json = urllib2.urlopen(req)#, timeout=1000*30)
    output_json = output_json.read()
    output_json = json.loads(output_json.decode('utf-8'))
    if "Loaded" not in output_json["status"]:
        pdb.gimp_message("Model could not be downloaded !")
        return False

    # Post request to API 
    post_json["text"] = text
    post_json["model"] = models[dropdown_index]
    post_json["image_shape"] = (layer.height, layer.width, layer.bpp)
    post_request_string = json.dumps(post_json).encode('utf-8')
    req =  urllib2.Request(url + "/run_inference", data=post_request_string) # this will make the method "POST"

    # Load output from json
    output_json = urllib2.urlopen(req)#, timeout=1000*30)
    output_json = output_json.read()
    output_json = json.loads(output_json.decode('utf-8'))
    output = output_json["image"]
    output_shape = output_json["image_shape"]

    # Load output image as layer
    rl = gimp.Layer(img, text, output_shape[1], output_shape[0], 0, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width,rl.height,True)
    region[:,:] = base64.b64decode(output)
    img.add_layer(rl,0)
    gimp.displays_flush()

    # Unmark undo
    img.undo_group_end()
    gimp.context_pop()
    return False


register(
    "python-fu-text-to-image",
    N_("Convert text to image"),
    "https://kritiksoman.github.io/GIMP-ML-Docs/docs-page.html#item-7-16",
    "Kritik Soman",
    "Dalle 3",
    "2024",
    N_("Text to image..."),
    "RGB*, GRAY*",
    [
        (PF_IMAGE, "image",       "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
	    (PF_STRING, "string", "Text", "photo of a cat"),
        (PF_OPTION, 'model', 'Model', 0, models)
    ],
    [],
    text_to_image,
    menu="<Image>/Layer/GIML-ML",
    domain=("gimp20-python", gimp.locale_directory)
    )

main()
