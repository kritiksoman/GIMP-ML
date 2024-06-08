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


url = 'http://127.0.0.1:8000'
post_json = {
    "pipeline": "text_to_image",
    "model": "model",
    "text": "TEXT",
    "source": "gimp2"
}

gettext.install("gimp20-python", gimp.locale_directory, unicode=True)

models = ["standard", "hd"]

def text_edit_image(img, layer, text, i, m):
    # Mark undo
    gimp.context_push()
    img.undo_group_start()

    # Load/set model
    post_json["pipeline"] = "text_edit_image"
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
    post_json["image"] = base64.b64encode(i.get_pixel_rgn(0, 0, i.width, i.height)[:, :])
    post_json["mask"] = base64.b64encode(m.get_pixel_rgn(0, 0, m.width, m.height)[:, :])
    post_json["image_shape"] = (i.height, i.width, i.bpp)
    post_json["mask_shape"] = (m.height, m.width, m.bpp)
    post_request_string = json.dumps(post_json).encode('utf-8')
    req =  urllib2.Request(url + "/run_inference", data=post_request_string) 

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
    "python-fu-text-edit-image",
    N_("Edit image with text prompt"),
    "https://kritiksoman.github.io/GIMP-ML-Docs/docs-page.html#item-7-16",
    "Kritik Soman",
    "Apache 2",
    "2024",
    N_("Edit image with text..."),
    "*",
    [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
	    (PF_STRING, "string", "Caption entire image with edit", "panda bear"),
        (PF_LAYER, "drawinglayer", "Original Image", None),
        (PF_LAYER, "drawinglayer", "Mask", None)
    ],
    [],
    text_edit_image,
    menu="<Image>/Layer/GIML-ML",
    domain=("gimp20-python", gimp.locale_directory)
    )

main()
