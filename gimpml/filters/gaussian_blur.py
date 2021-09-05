#!/usr/bin/env python3
# coding: utf-8

# def add_gaussian_blur(image_path, save_path):
try:
    import os
    import pickle
    import gi
    gi.require_version("Gimp", "3.0")
    from gi.repository import Gimp, GObject, Gio

    install_location = os.path.join(os.path.expanduser("~"), "GIMP-ML")
    with open(os.path.join(install_location, "gimp_ml_augment.pkl"), "rb") as file:
        data_output = pickle.load(file)

    image_path = data_output['image_path']
    save_path = data_output['save_path']
    horizontal = data_output['horizontal']
    vertical = data_output['vertical']
    method = data_output['method']

    image = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.file_new_for_path(image_path))  # image
    image_layer = image.get_active_layer()  # drawable

    # run plugin
    Gimp.get_pdb().run_procedure(
        "plug-in-gauss",
        [
            GObject.Value(Gimp.RunMode, Gimp.RunMode.NONINTERACTIVE),
            GObject.Value(Gimp.Image, image),
            GObject.Value(Gimp.Drawable, image_layer),
            GObject.Value(GObject.TYPE_DOUBLE, horizontal),
            GObject.Value(GObject.TYPE_DOUBLE, vertical),
            GObject.Value(GObject.TYPE_INT, method),
        ]
    )
    image_layer = image.get_active_layer()

    # save
    interlace, compression = 0, 2
    Gimp.get_pdb().run_procedure(
        "file-png-save",
        [
            GObject.Value(Gimp.RunMode, Gimp.RunMode.NONINTERACTIVE),
            GObject.Value(Gimp.Image, image),
            GObject.Value(GObject.TYPE_INT, 1),
            GObject.Value(
                Gimp.ObjectArray, Gimp.ObjectArray.new(Gimp.Drawable, [image_layer], 0)
            ),
            GObject.Value(
                Gio.File,
                Gio.File.new_for_path(save_path),
            ),
            GObject.Value(GObject.TYPE_BOOLEAN, interlace),
            GObject.Value(GObject.TYPE_INT, compression),
            # write all PNG chunks except oFFs(ets)
            GObject.Value(GObject.TYPE_BOOLEAN, True),
            GObject.Value(GObject.TYPE_BOOLEAN, True),
            GObject.Value(GObject.TYPE_BOOLEAN, False),
            GObject.Value(GObject.TYPE_BOOLEAN, True),
        ],
    )

    # quit
    Gimp.get_pdb().run_procedure("gimp-quit", [GObject.Value(GObject.TYPE_BOOLEAN, True)])

except:
    pass
# gimp-2.99 -idf --batch-interpreter=python-fu-eval -b - < gaussian_blur.py
