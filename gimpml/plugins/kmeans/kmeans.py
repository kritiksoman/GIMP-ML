#!/usr/bin/env python3
# coding: utf-8

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
Performs k means clustering for current layer.
"""

import csv
import math
import sys

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
import os
import pickle
import subprocess

_ = gettext.gettext


def N_(message): return message


class StringEnum:
    """
    Helper class for when you want to use strings as keys of an enum. The values would be
    user facing strings that might undergo translation.

    The constructor accepts an even amount of arguments. Each pair of arguments
    is a key/value pair.
    """

    def __init__(self, *args):
        self.keys = []
        self.values = []

        for i in range(len(args) // 2):
            self.keys.append(args[i * 2])
            self.values.append(args[i * 2 + 1])

    def get_tree_model(self):
        """ Get a tree model that can be used in GTK widgets. """
        tree_model = Gtk.ListStore(GObject.TYPE_STRING, GObject.TYPE_STRING)
        for i in range(len(self.keys)):
            tree_model.append([self.keys[i], self.values[i]])
        return tree_model

    def __getattr__(self, name):
        """ Implements access to the key. For example, if you provided a key "red", then you could access it by
            referring to
               my_enum.red
            It may seem silly as "my_enum.red" is longer to write then just "red",
            but this provides verification that the key is indeed inside enum. """
        key = name.replace("_", " ")
        if key in self.keys:
            return key
        raise AttributeError("No such key string " + key)


# output_format_enum = StringEnum(
#     "pixel count", _("Pixel count"),
#     "normalized", _("Normalized"),
#     "percent", _("Percent")
# )


def k_means(procedure, image, drawable, n_cluster, position, progress_bar):
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "tools")
    with open(os.path.join(config_path, 'gimp_ml_config.pkl'), 'rb') as file:
        data_output = pickle.load(file)
    weight_path = data_output["weight_path"]
    python_path = data_output["python_path"]
    plugin_path = os.path.join(config_path, 'kmeans.py')

    Gimp.context_push()
    image.undo_group_start()

    interlace, compression = 0, 2
    Gimp.get_pdb().run_procedure('file-png-save', [
        GObject.Value(Gimp.RunMode, Gimp.RunMode.NONINTERACTIVE),
        GObject.Value(Gimp.Image, image),
        GObject.Value(GObject.TYPE_INT, 1),
        GObject.Value(Gimp.ObjectArray, Gimp.ObjectArray.new(Gimp.Drawable, drawable, 1)),
        GObject.Value(Gio.File, Gio.File.new_for_path(os.path.join(weight_path, '..', 'cache.png'))),
        GObject.Value(GObject.TYPE_BOOLEAN, interlace),
        GObject.Value(GObject.TYPE_INT, compression),
        # write all PNG chunks except oFFs(ets)
        GObject.Value(GObject.TYPE_BOOLEAN, True),
        GObject.Value(GObject.TYPE_BOOLEAN, True),
        GObject.Value(GObject.TYPE_BOOLEAN, False),
        GObject.Value(GObject.TYPE_BOOLEAN, True),
    ])

    with open(os.path.join(weight_path, '..', 'gimp_ml_run.pkl'), 'wb') as file:
        pickle.dump({"n_cluster": int(n_cluster), "position": bool(position)}, file)

    subprocess.call([python_path, plugin_path])

    result = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.file_new_for_path(os.path.join(weight_path, '..', 'cache.png')))
    result_layer = result.get_active_layer()
    copy = Gimp.Layer.new_from_drawable(result_layer, image)
    copy.set_name("K Means")
    copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)#DIFFERENCE_LEGACY
    image.insert_layer(copy, None, -1)

    image.undo_group_end()
    Gimp.context_pop()

    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


def run(procedure, run_mode, image, n_drawables, layer, args, data):
    # gio_file = args.index(0)
    n_cluster = args.index(0)
    position = args.index(1)
    # output_format = args.index(3)
    # force_cpu = args.index(2)

    progress_bar = None
    config = None

    if run_mode == Gimp.RunMode.INTERACTIVE:

        config = procedure.create_config()

        # Set properties from arguments. These properties will be changed by the UI.
        # config.set_property("file", gio_file)
        # config.set_property("n_cluster", n_cluster)
        # config.set_property("sample_average", sample_average)
        # config.set_property("output_format", output_format)
        # config.set_property("force_cpu", force_cpu)
        config.begin_run(image, run_mode, args)

        GimpUi.init("kmeans.py")
        use_header_bar = Gtk.Settings.get_default().get_property("gtk-dialogs-use-header")
        dialog = GimpUi.Dialog(use_header_bar=use_header_bar,
                               title=_("K Means..."))
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

        # UI for the file parameter

        # def choose_file(widget):
        #     if file_chooser_dialog.run() == Gtk.ResponseType.OK:
        #         if file_chooser_dialog.get_file() is not None:
        #             config.set_property("file", file_chooser_dialog.get_file())
        #             file_entry.set_text(file_chooser_dialog.get_file().get_path())
        #     file_chooser_dialog.hide()
        #
        # file_chooser_button = Gtk.Button.new_with_mnemonic(label=_("_File..."))
        # grid.attach(file_chooser_button, 0, 0, 1, 1)
        # file_chooser_button.show()
        # file_chooser_button.connect("clicked", choose_file)
        #
        # file_entry = Gtk.Entry.new()
        # grid.attach(file_entry, 1, 0, 1, 1)
        # file_entry.set_width_chars(40)
        # file_entry.set_placeholder_text(_("Choose export file..."))
        # # if gio_file is not None:
        # #     file_entry.set_text(gio_file.get_path())
        # file_entry.show()
        #
        # file_chooser_dialog = Gtk.FileChooserDialog(use_header_bar=use_header_bar,
        #                                             title=_("Histogram Export file..."),
        #                                             action=Gtk.FileChooserAction.SAVE)
        # file_chooser_dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        # file_chooser_dialog.add_button("_OK", Gtk.ResponseType.OK)

        # n_cluster parameter
        label = Gtk.Label.new_with_mnemonic(_("_Clusters"))
        grid.attach(label, 0, 1, 1, 1)
        label.show()
        spin = GimpUi.prop_spin_button_new(config, "n_cluster", step_increment=1, page_increment=10, digits=0)
        grid.attach(spin, 1, 1, 1, 1)
        spin.show()

        # Sample average parameter
        spin = GimpUi.prop_check_button_new(config, "position", _("Use _Position"))
        spin.set_tooltip_text(_("If checked, x, y coordinates will be used as features for k means clustering"))
        grid.attach(spin, 1, 2, 1, 1)
        spin.show()

        # # Output format parameter
        # label = Gtk.Label.new_with_mnemonic(_("_Output Format"))
        # grid.attach(label, 0, 3, 1, 1)
        # label.show()
        # combo = GimpUi.prop_string_combo_box_new(config, "output_format", output_format_enum.get_tree_model(), 0, 1)
        # grid.attach(combo, 1, 3, 1, 1)
        # combo.show()

        # # Force CPU parameter
        # spin = GimpUi.prop_check_button_new(config, "force_cpu", _("Force _CPU"))
        # spin.set_tooltip_text(_("If checked, CPU is used for model inference."
        #                         " Otherwise, GPU will be used if available."))
        # grid.attach(spin, 1, 3, 1, 1)
        # spin.show()

        progress_bar = Gtk.ProgressBar()
        vbox.add(progress_bar)
        progress_bar.show()

        dialog.show()
        if dialog.run() != Gtk.ResponseType.OK:
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL,
                                               GLib.Error())

        # Extract values from UI
        # gio_file = Gio.file_new_for_path(file_entry.get_text())  # config.get_property("file")
        n_cluster = config.get_property("n_cluster")
        position = config.get_property("position")
        # force_cpu = config.get_property("force_cpu")

    # if gio_file is None:
    #     error = 'No file given'
    #     return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR,
    #                                        GLib.Error(error))

    result = k_means(procedure, image, layer,
                              n_cluster, position, progress_bar)

    # If the execution was successful, save parameters so they will be restored next time we show dialog.
    if result.index(0) == Gimp.PDBStatusType.SUCCESS and config is not None:
        config.end_run(Gimp.PDBStatusType.SUCCESS)

    return result


class Kmeans(Gimp.PlugIn):
    ## Parameters ##
    __gproperties__ = {
        # "filename": (str,
        #              # TODO: I wanted this property to be a path (and not just str) , so I could use
        #              # prop_file_chooser_button_new to open a file dialog. However, it fails without an error message.
        #              # Gimp.ConfigPath,
        #              _("Histogram _File"),
        #              _("Histogram _File"),
        #              "k_means.csv",
        #              # Gimp.ConfigPathType.FILE,
        #              GObject.ParamFlags.READWRITE),
        # "file": (Gio.File,
        #          _("Histogram _File"),
        #          "Histogram export file",
        #          GObject.ParamFlags.READWRITE),
        "n_cluster": (float,
                  _("_Clusters"),
                  "Number of clusters",
                  3, 64, 5,
                  GObject.ParamFlags.READWRITE),
        "position": (bool,
                   _("Use _Position"),
                   "Use as position of pixels",
                   False,
                   GObject.ParamFlags.READWRITE),
        # "output_format": (str,
        #                   _("Output format"),
        #                   "Output format: 'pixel count', 'normalized', 'percent'",
        #                   "pixel count",
        #                   GObject.ParamFlags.READWRITE),
        # "force_cpu": (bool,
        #               _("Force _CPU"),
        #               "Force CPU",
        #               False,
        #               GObject.ParamFlags.READWRITE),
    }

    ## GimpPlugIn virtual methods ##
    def do_query_procedures(self):
        self.set_translation_domain("gimp30-python",
                                    Gio.file_new_for_path(Gimp.locale_directory()))
        return ['kmeans']

    def do_create_procedure(self, name):
        procedure = None
        if name == 'kmeans':
            procedure = Gimp.ImageProcedure.new(self, name,
                                                Gimp.PDBProcType.PLUGIN,
                                                run, None)

            procedure.set_image_types("*")
            procedure.set_documentation(
                N_("Performs k means clustering for current layer."),
                globals()["__doc__"],  # This includes the docstring, on the top of the file
                name)
            procedure.set_menu_label(N_("_K Means..."))
            procedure.set_attribution("João S. O. Bueno",
                                      "(c) GPL V3.0 or later",
                                      "2014")
            procedure.add_menu_path("<Image>/Layer/GIMP-ML/")

            # procedure.add_argument_from_property(self, "file")
            procedure.add_argument_from_property(self, "n_cluster")
            procedure.add_argument_from_property(self, "position")
            # procedure.add_argument_from_property(self, "output_format")
            # procedure.add_argument_from_property(self, "force_cpu")

        return procedure


Gimp.main(Kmeans.__gtype__, sys.argv)
