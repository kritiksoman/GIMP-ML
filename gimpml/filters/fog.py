#!/usr/bin/env python3
# coding: utf-8
try:
    import time
    import os
    import pickle
    import gi

    gi.require_version("Gimp", "3.0")
    from gi.repository import Gimp, GObject, Gio

    install_location = os.path.join(os.path.expanduser("~"), "GIMP-ML")
    with open(os.path.join(install_location, "gimp_ml_augment.pkl"), "rb") as file:
        data_output = pickle.load(file)

    image_path = data_output['image_path']
    opacity = data_output['opacity']
    rgb = data_output['rgb']
    save_path = data_output['save_path']
    turbulence = data_output['turbulence']

    image = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.file_new_for_path(image_path))  # image
    image_layer = image.get_active_layer()

    if image.get_base_type() is Gimp.ImageBaseType.RGB:
        type = Gimp.ImageType.RGBA_IMAGE
    else:
        type = Gimp.ImageType.GRAYA_IMAGE

    # fog = Gimp.Layer.new_from_drawable(image_layer, image)
    fog = Gimp.Layer.new(image, "tmp",
                         image_layer.get_width(), image_layer.get_height(),
                         type, opacity,
                         Gimp.LayerMode.NORMAL)
    fog.fill(Gimp.FillType.TRANSPARENT)
    image.insert_layer(fog, image_layer.get_parent(), image.get_item_position(image_layer))
    color = Gimp.RGB()
    color.set(rgb[0], rgb[1], rgb[2])
    Gimp.context_set_background(color)
    fog.edit_fill(Gimp.FillType.BACKGROUND)

    # create a layer mask for the new layer
    mask = fog.create_mask(0)
    fog.add_mask(mask)

    # add some clouds to the layer
    Gimp.get_pdb().run_procedure('plug-in-plasma', [
        GObject.Value(Gimp.RunMode, Gimp.RunMode.NONINTERACTIVE),
        GObject.Value(Gimp.Image, image),
        GObject.Value(Gimp.Drawable, mask),
        GObject.Value(GObject.TYPE_INT, int(time.time())),
        GObject.Value(GObject.TYPE_DOUBLE, turbulence),
    ])
    # apply the clouds to the layer
    fog.remove_mask(Gimp.MaskApplyMode.APPLY)
    fog.set_visible(True)

    thumb = image.duplicate()
    layer = thumb.merge_visible_layers(Gimp.MergeType.CLIP_TO_IMAGE)

    # save
    interlace, compression = 0, 2
    Gimp.get_pdb().run_procedure(
        "file-png-save",
        [
            GObject.Value(Gimp.RunMode, Gimp.RunMode.NONINTERACTIVE),
            GObject.Value(Gimp.Image, thumb),
            GObject.Value(GObject.TYPE_INT, 1),
            GObject.Value(
                Gimp.ObjectArray, Gimp.ObjectArray.new(Gimp.Drawable, [layer], 0)
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
