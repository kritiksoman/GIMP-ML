#!/usr/bin/env python3
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


Object detection on the current layer.
"""
import gi

gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gimp, GimpUi, GObject, GLib, Gio, Gtk
import gettext
import subprocess
import pickle
import os
import sys

sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")])
from plugin_utils import *

_ = gettext.gettext
image_paths = {
    "logo": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "images", "plugin_logo.png"
    ),
    "error": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "images", "error_icon.png"
    ),
}


def filterfolder(procedure, args_dict, config_path_output):
    # Save inference parameters and layers
    weight_path = config_path_output["weight_path"]
    python_path = config_path_output["python_path"]
    plugin_path = config_path_output["plugin_path"]
    args_dict["inference_status"] = "started"
    args_dict["get_predict_image"] = False
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "wb") as file:
        pickle.dump(args_dict, file)

    # Run inference and load as layer
    subprocess.call([python_path, plugin_path])
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    if data_output["inference_status"] == "success":
        count = data_output["count"]
        if count == 0:
            message = "No files found with entered objects."
        elif count == 1:
            message = str(count) + " file moved to " + os.path.join(args_dict["image_path"], "filtered")
        else:
            message = str(count) + " files moved to " + os.path.join(args_dict["image_path"], "filtered")
        show_dialog(
            message,
            "Inference Complete.",
            "logo",
            image_paths
        )

        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    else:
        show_dialog(
            "Inference not successful. See error_log.txt in GIMP-ML folder.",
            "Error !",
            "error",
            image_paths
        )
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


def run(procedure, run_mode, args):
    args_dict = {}

    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "..", "tools"
    )
    with open(os.path.join(config_path, "gimp_ml_config.pkl"), "rb") as file:
        config_path_output = pickle.load(file)
    python_path = config_path_output["python_path"]
    config_path_output["plugin_path"] = os.path.join(config_path, "detectobjects.py")

    config = procedure.create_config()

    GimpUi.init("filterfolder.py")
    use_header_bar = Gtk.Settings.get_default().get_property(
        "gtk-dialogs-use-header"
    )
    dialog = GimpUi.Dialog(use_header_bar=use_header_bar, title=_("Filter folder..."))
    dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
    dialog.add_button("_Help", Gtk.ResponseType.APPLY)
    dialog.add_button("_Run Inference", Gtk.ResponseType.OK)

    vbox = Gtk.Box(
        orientation=Gtk.Orientation.VERTICAL, homogeneous=False, spacing=10
    )
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
    def choose_file(widget):
        if file_chooser_dialog.run() == Gtk.ResponseType.OK:
            if file_chooser_dialog.get_file() is not None:
                # config.set_property("file", file_chooser_dialog.get_file())
                file_entry.set_text(file_chooser_dialog.get_file().get_path())
        file_chooser_dialog.hide()

    file_chooser_button = Gtk.Button.new_with_mnemonic(label=_("_Folder..."))
    grid.attach(file_chooser_button, 0, 0, 1, 1)
    file_chooser_button.show()
    file_chooser_button.connect("clicked", choose_file)

    file_entry = Gtk.Entry.new()
    grid.attach(file_entry, 1, 0, 1, 1)
    file_entry.set_width_chars(40)
    file_entry.set_placeholder_text(_("Choose export folder..."))
    file_entry.set_text(os.path.join(os.path.expanduser("~"), "Pictures"))
    file_entry.show()

    file_chooser_dialog = Gtk.FileChooserDialog(
        use_header_bar=use_header_bar,
        title=_("Frame Export folder..."),
        action=Gtk.FileChooserAction.SELECT_FOLDER,
    )
    file_chooser_dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
    file_chooser_dialog.add_button("_OK", Gtk.ResponseType.OK)

    # Filter Objects
    filter_objects = Gtk.Entry.new()
    grid.attach(filter_objects, 1, 1, 1, 1)
    filter_objects.set_width_chars(40)
    filter_objects.set_text("Person|Cars")
    filter_objects.show()

    filer_objects_text = _("Objects to search")
    filter_label = Gtk.Label(label=filer_objects_text)
    grid.attach(filter_label, 0, 1, 1, 1)
    vbox.pack_start(filter_label, False, False, 1)
    filter_label.show()

    # Show Logo
    logo = Gtk.Image.new_from_file(image_paths["logo"])
    vbox.pack_start(logo, False, False, 1)
    logo.show()

    # Show Custom Text
    license_text = _("For complete list of objects see help.")
    label = Gtk.Label(label=license_text)
    vbox.pack_start(label, False, False, 1)
    label.show()

    # Show License
    license_text = _("PLUGIN LICENSE : Apache-2.0")
    label = Gtk.Label(label=license_text)
    vbox.pack_start(label, False, False, 1)
    label.show()

    progress_bar = Gtk.ProgressBar()
    vbox.add(progress_bar)
    progress_bar.show()

    # Wait for user to click
    dialog.show()
    while True:
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            args_dict["image_path"] = file_entry.get_text()
            args_dict["objects"] = filter_objects.get_text()
            result = filterfolder(
                procedure, args_dict, config_path_output
            )
            if result.index(0) == Gimp.PDBStatusType.SUCCESS and config is not None:
                config.end_run(Gimp.PDBStatusType.SUCCESS)
            return result
        elif response == Gtk.ResponseType.APPLY:
            url = "https://kritiksoman.github.io/GIMP-ML-Docs/docs-page.html#item-7-15"
            Gio.app_info_launch_default_for_uri(url, None)
            continue
        else:
            dialog.destroy()
            return procedure.new_return_values(
                Gimp.PDBStatusType.CANCEL, GLib.Error()
            )


class FilterFolder(Gimp.PlugIn):

    @GObject.Property(type=Gimp.RunMode,
                      default=Gimp.RunMode.NONINTERACTIVE,
                      nick="Run mode", blurb="The run mode")
    def run_mode(self):
        """Read-write integer property."""
        return self.runmode

    @run_mode.setter
    def run_mode(self, runmode):
        self.runmode = runmode

    ## GimpPlugIn virtual methods ##
    def do_query_procedures(self):
        self.set_translation_domain(
            "gimp30-python", Gio.file_new_for_path(Gimp.locale_directory())
        )
        return ["filterfolder"]

    def do_create_procedure(self, name):
        procedure = None
        if name == "filterfolder":
            procedure = Gimp.Procedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, run, None
            )
            # procedure.set_image_types("*")
            procedure.set_documentation(
                N_("Detects objects on the current layer."),
                globals()[
                    "__doc__"
                ],  # This includes the docstring, on the top of the file
                name,
            )
            procedure.set_menu_label(N_("Filter Folder Objects..."))
            procedure.set_attribution("Kritik Soman", "GIMP-ML", "2021")
            procedure.add_menu_path("<Image>/Layer/GIMP-ML/")
            procedure.add_argument_from_property(self, "run-mode")

        return procedure


Gimp.main(FilterFolder.__gtype__, sys.argv)
