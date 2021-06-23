#!/usr/bin/env python3
#coding: utf-8
"""
 .d8888b.  8888888 888b     d888 8888888b.       888b     d888 888
d88P  Y88b   888   8888b   d8888 888   Y88b      8888b   d8888 888
888    888   888   88888b.d88888 888    888      88888b.d88888 888
888          888   888Y88888P888 888   d88P      888Y88888P888 888
888  88888   888   888 Y888P 888 8888888P"       888 Y888P 888 888
888    888   888   888  Y8P  888 888             888  Y8P  888 888
Y88b  d88P   888   888   "   888 888             888   "   888 888
 "Y8888P88 8888888 888       888 888             888       888 88888888


Opens the color palette as a new image file in GIMP.
"""
import gi

gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
from gi.repository import GObject
from gi.repository import GLib
from gi.repository import Gio
import time
import sys
import os
import gettext

_ = gettext.gettext


def N_(message): return message


def colorpalette(procedure, run_mode, image, n_drawables, drawable, args, data):
    image_new = Gimp.Image.new(1200, 675, 0)  # 0 for RGB
    display = Gimp.Display.new(image_new)


    result = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.file_new_for_path(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'color_palette.png')))
    result_layer = result.get_active_layer()
    copy = Gimp.Layer.new_from_drawable(result_layer, image_new)
    copy.set_name("Color Palette")
    copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)# DIFFERENCE_LEGACY
    image_new.insert_layer(copy, None, -1)
    Gimp.displays_flush()


    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


class ColorPalette(Gimp.PlugIn):
    ## Parameters ##
    __gproperties__ = {
        # "name": (str,
        #          _("Layer name"),
        #          _("Layer name"),
        #          _("Clouds"),
        #          GObject.ParamFlags.READWRITE),
        # "color": (Gimp.RGB,
        #           _("Fog color"),
        #           _("Fog color"),
        #           GObject.ParamFlags.READWRITE),
        # "turbulence": (float,
        #                _("Turbulence"),
        #                _("Turbulence"),
        #                0.0, 10.0, 1.0,
        #                GObject.ParamFlags.READWRITE),
        # "opacity": (float,
        #             _("Opacity"),
        #             _("Opacity"),
        #             0.0, 100.0, 100.0,
        #             GObject.ParamFlags.READWRITE),
    }

    ## GimpPlugIn virtual methods ##
    def do_query_procedures(self):
        self.set_translation_domain("gimp30-python",
                                    Gio.file_new_for_path(Gimp.locale_directory()))

        return ['colorpalette']

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(self, name,
                                            Gimp.PDBProcType.PLUGIN,
                                            colorpalette, None)
        procedure.set_image_types("RGB*, GRAY*");
        procedure.set_documentation(N_("Add a layer of fog"),
                                    "Adds a layer of fog to the image.",
                                    name)
        procedure.set_menu_label(N_("_Color Palette..."))
        procedure.set_attribution("Kritik Soman",
                                  "GIMP-ML",
                                  "2021")
        procedure.add_menu_path("<Image>/Layer/GIMP-ML/")

        # procedure.add_argument_from_property(self, "name")
        # TODO: add support for GBoxed values.
        # procedure.add_argument_from_property(self, "color")
        # procedure.add_argument_from_property(self, "turbulence")
        # procedure.add_argument_from_property(self, "opacity")
        return procedure


Gimp.main(ColorPalette.__gtype__, sys.argv)