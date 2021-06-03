#!/usr/bin/env python3
#coding: utf-8

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Extracts the monocular depth of the current layer.
"""
import pickle
import csv
import math
import sys
import time
import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
gi.require_version('GimpUi', '3.0')
from gi.repository import GimpUi
from gi.repository import GObject
from gi.repository import GLib
from gi.repository import Gio
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import gettext
_ = gettext.gettext
def N_(message): return message

import subprocess
import pickle
import os

def monodepth(procedure, image, drawable, force_cpu, progress_bar):

    #
    # install_location = os.path.join(os.path.expanduser("~"), "GIMP-ML")
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "tools")
    with open(os.path.join(config_path, 'gimp_ml_config.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    weight_path = data_output["weight_path"]
    python_path = data_output["python_path"]
    # python_path = r"D:/PycharmProjects/gimpenv3/Scripts/python.exe"
    plugin_path = os.path.join(config_path, 'monodepth.py')

    # plugin_path = r"D:/PycharmProjects/GIMP3-ML-pip/gimpml/tools/monodepth.py"

    Gimp.context_push()
    image.undo_group_start()

    interlace, compression = 0, 2
    Gimp.get_pdb().run_procedure('file-png-save', [
        GObject.Value(Gimp.RunMode, Gimp.RunMode.NONINTERACTIVE),
        GObject.Value(Gimp.Image, image),
        GObject.Value(GObject.TYPE_INT, 1),
        GObject.Value(Gimp.ObjectArray, Gimp.ObjectArray.new(Gimp.Drawable, [drawable], 1)),
        GObject.Value(Gio.File, Gio.File.new_for_path(os.path.join(weight_path, 'cache.png'))),
        GObject.Value(GObject.TYPE_BOOLEAN, interlace),
        GObject.Value(GObject.TYPE_INT, compression),
        # write all PNG chunks except oFFs(ets)
        GObject.Value(GObject.TYPE_BOOLEAN, True),
        GObject.Value(GObject.TYPE_BOOLEAN, True),
        GObject.Value(GObject.TYPE_BOOLEAN, False),
        GObject.Value(GObject.TYPE_BOOLEAN, True),
    ])

    subprocess.call([python_path, plugin_path])

    result = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.file_new_for_path(os.path.join(weight_path, 'cache.png')))
    result_layer = result.get_active_layer()
    copy = Gimp.Layer.new_from_drawable(result_layer, image)
    copy.set_name("Mono Depth")
    copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)#DIFFERENCE_LEGACY
    image.insert_layer(copy, None, -1)

    image.undo_group_end()
    Gimp.context_pop()

    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())



def run(procedure, run_mode, image, layer, args, data):
    # gio_file = args.index(0)
    # bucket_size = args.index(0)
    force_cpu = args.index(1)
    # output_format = args.index(2)

    progress_bar = None
    config = None

    if run_mode == Gimp.RunMode.INTERACTIVE:

        config = procedure.create_config()

        # Set properties from arguments. These properties will be changed by the UI.
        #config.set_property("file", gio_file)
        #config.set_property("bucket_size", bucket_size)
        config.set_property("force_cpu", force_cpu)
        #config.set_property("output_format", output_format)
        config.begin_run(image, run_mode, args)

        GimpUi.init("monodepth.py")
        use_header_bar = Gtk.Settings.get_default().get_property("gtk-dialogs-use-header")
        dialog = GimpUi.Dialog(use_header_bar=use_header_bar,
                             title=_("Mono Depth..."))
        dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("_OK", Gtk.ResponseType.OK)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL,
                       homogeneous=False, spacing=10)
        dialog.get_content_area().add(vbox)
        vbox.show()

        # Create grid to set all the properties inside.
        grid = Gtk.Grid()
        grid.set_column_homogeneous(False)
        grid.set_border_width(10)
        grid.set_column_spacing(10)
        grid.set_row_spacing(10)
        vbox.add(grid)
        grid.show()

        # # Bucket size parameter
        # label = Gtk.Label.new_with_mnemonic(_("_Bucket Size"))
        # grid.attach(label, 0, 1, 1, 1)
        # label.show()
        # spin = GimpUi.prop_spin_button_new(config, "bucket_size", step_increment=0.001, page_increment=0.1, digits=3)
        # grid.attach(spin, 1, 1, 1, 1)
        # spin.show()

        # Force CPU parameter
        spin = GimpUi.prop_check_button_new(config, "force_cpu", _("Force _CPU"))
        spin.set_tooltip_text(_("If checked, CPU is used for model inference."
                                " Otherwise, GPU will be used if available."))
        grid.attach(spin, 1, 2, 1, 1)
        spin.show()

        # # Output format parameter
        # label = Gtk.Label.new_with_mnemonic(_("_Output Format"))
        # grid.attach(label, 0, 3, 1, 1)
        # label.show()
        # combo = GimpUi.prop_string_combo_box_new(config, "output_format", output_format_enum.get_tree_model(), 0, 1)
        # grid.attach(combo, 1, 3, 1, 1)
        # combo.show()

        progress_bar = Gtk.ProgressBar()
        vbox.add(progress_bar)
        progress_bar.show()

        dialog.show()
        if dialog.run() != Gtk.ResponseType.OK:
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL,
                                               GLib.Error())

    result = monodepth(procedure, image, layer, force_cpu, progress_bar)

    # If the execution was successful, save parameters so they will be restored next time we show dialog.
    if result.index(0) == Gimp.PDBStatusType.SUCCESS and config is not None:
        config.end_run(Gimp.PDBStatusType.SUCCESS)

    return result


class MonoDepth(Gimp.PlugIn):

    ## Parameters ##
    __gproperties__ = {
        # "filename": (str,
        #              # TODO: I wanted this property to be a path (and not just str) , so I could use
        #              # prop_file_chooser_button_new to open a file dialog. However, it fails without an error message.
        #              # Gimp.ConfigPath,
        #              _("Histogram _File"),
        #              _("Histogram _File"),
        #              "monodepth.csv",
        #              # Gimp.ConfigPathType.FILE,
        #              GObject.ParamFlags.READWRITE),
        # "file": (Gio.File,
        #          _("Histogram _File"),
        #          "Histogram export file",
        #          GObject.ParamFlags.READWRITE),
        # "bucket_size":  (float,
        #                  _("_Bucket Size"),
        #                  "Bucket Size",
        #                  0.001, 1.0, 0.01,
        #                  GObject.ParamFlags.READWRITE),
        "force_cpu": (bool,
                           _("Force _CPU"),
                           "Force CPU",
                           False,
                           GObject.ParamFlags.READWRITE),
        # "output_format": (str,
        #                   _("Output format"),
        #                   "Output format: 'pixel count', 'normalized', 'percent'",
        #                   "pixel count",
        #                   GObject.ParamFlags.READWRITE),
    }

    ## GimpPlugIn virtual methods ##
    def do_query_procedures(self):
        self.set_translation_domain("gimp30-python",
                                    Gio.file_new_for_path(Gimp.locale_directory()))
        return ['monodepth']

    def do_create_procedure(self, name):
        procedure = None
        if name == 'monodepth':
            procedure = Gimp.ImageProcedure.new(self, name, Gimp.PDBProcType.PLUGIN, run, None)
            procedure.set_image_types("*")
            procedure.set_documentation (
                N_("Extracts the monocular depth of the current layer."),
                globals()["__doc__"],  # This includes the docstring, on the top of the file
                name)
            procedure.set_menu_label(N_("_Mono Depth..."))
            procedure.set_attribution("Kritik Soman",
                                      "GIMP-ML",
                                      "2021")
            procedure.add_menu_path("<Image>/Layer/GIMP-ML/")

            # procedure.add_argument_from_property(self, "file")
            # procedure.add_argument_from_property(self, "bucket_size")
            procedure.add_argument_from_property(self, "force_cpu")
            # procedure.add_argument_from_property(self, "output_format")

        return procedure


Gimp.main(MonoDepth.__gtype__, sys.argv)
