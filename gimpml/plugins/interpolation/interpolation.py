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


Performs interpolation between two layers and exports results to a folder.
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
    "colorpalette": os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "colorpalette",
        "color_palette.png",
    ),
    "logo": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "images", "plugin_logo.png"
    ),
    "error": os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "images", "error_icon.png"
    ),
}


def interpolation(
    procedure,
    image,
    n_drawables,
    drawables,
    force_cpu,
    progress_bar,
    gio_file,
    config_path_output,
):
    # Save inference parameters and layers
    weight_path = config_path_output["weight_path"]
    python_path = config_path_output["python_path"]
    plugin_path = config_path_output["plugin_path"]

    Gimp.context_push()
    image.undo_group_start()

    for index, drawable in enumerate(drawables):
        interlace, compression = 0, 2
        Gimp.get_pdb().run_procedure(
            "file-png-save",
            [
                GObject.Value(Gimp.RunMode, Gimp.RunMode.NONINTERACTIVE),
                GObject.Value(Gimp.Image, image),
                GObject.Value(GObject.TYPE_INT, 1),
                GObject.Value(
                    Gimp.ObjectArray, Gimp.ObjectArray.new(Gimp.Drawable, [drawable], 0)
                ),
                GObject.Value(
                    Gio.File,
                    Gio.File.new_for_path(
                        os.path.join(weight_path, "..", "cache" + str(index) + ".png")
                    ),
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

    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "wb") as file:
        pickle.dump(
            {
                "force_cpu": bool(force_cpu),
                "gio_file": str(gio_file),
                "inference_status": "started",
            },
            file,
        )

    # Run inference and load as layer
    subprocess.call([python_path, plugin_path])
    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "rb") as file:
        data_output = pickle.load(file)
    image.undo_group_end()
    Gimp.context_pop()
    if data_output["inference_status"] == "success":
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())

    else:
        show_dialog(
            "Inference not successful. See error_log.txt in GIMP-ML folder.",
            "Error !",
            "error",
            image_paths
        )
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


def run(procedure, run_mode, image, n_drawables, layer, args, data):
    gio_file = args.index(0)
    force_cpu = args.index(1)

    progress_bar = None
    config = None

    if run_mode == Gimp.RunMode.INTERACTIVE:
        # Get all paths
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "tools"
        )
        with open(os.path.join(config_path, "gimp_ml_config.pkl"), "rb") as file:
            config_path_output = pickle.load(file)
        python_path = config_path_output["python_path"]
        config_path_output["plugin_path"] = os.path.join(
            config_path, "interpolation.py"
        )

        config = procedure.create_config()
        config.set_property("force_cpu", force_cpu)
        config.begin_run(image, run_mode, args)

        GimpUi.init("interpolation.py")
        use_header_bar = Gtk.Settings.get_default().get_property(
            "gtk-dialogs-use-header"
        )
        dialog = GimpUi.Dialog(
            use_header_bar=use_header_bar, title=_("interpolation...")
        )
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

        # Show Logo
        logo = Gtk.Image.new_from_file(image_paths["logo"])
        # grid.attach(logo, 0, 0, 1, 1)
        vbox.pack_start(logo, False, False, 1)
        logo.show()

        # Show License
        license_text = _("PLUGIN LICENSE : MIT")
        label = Gtk.Label(label=license_text)
        # grid.attach(label, 1, 1, 1, 1)
        vbox.pack_start(label, False, False, 1)
        label.show()

        # UI for the file parameter
        def choose_file(widget):
            if file_chooser_dialog.run() == Gtk.ResponseType.OK:
                if file_chooser_dialog.get_file() is not None:
                    config.set_property("file", file_chooser_dialog.get_file())
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
        if gio_file is not None:
            file_entry.set_text(gio_file.get_path())
        file_entry.show()

        file_chooser_dialog = Gtk.FileChooserDialog(
            use_header_bar=use_header_bar,
            title=_("Frame Export folder..."),
            action=Gtk.FileChooserAction.SELECT_FOLDER,
        )
        file_chooser_dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        file_chooser_dialog.add_button("_OK", Gtk.ResponseType.OK)

        # Force CPU parameter
        spin = GimpUi.prop_check_button_new(config, "force_cpu", _("Force _CPU"))
        spin.set_tooltip_text(
            _(
                "If checked, CPU is used for model inference."
                " Otherwise, GPU will be used if available."
            )
        )
        grid.attach(spin, 1, 2, 1, 1)
        spin.show()

        progress_bar = Gtk.ProgressBar()
        vbox.add(progress_bar)
        progress_bar.show()

        # Wait for user to click
        dialog.show()
        while True:
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                force_cpu = config.get_property("force_cpu")
                gio_file = file_entry.get_text()
                result = interpolation(
                    procedure,
                    image,
                    n_drawables,
                    layer,
                    force_cpu,
                    progress_bar,
                    gio_file,
                    config_path_output,
                )
                # If the execution was successful, save parameters so they will be restored next time we show dialog.
                if result.index(0) == Gimp.PDBStatusType.SUCCESS and config is not None:
                    config.end_run(Gimp.PDBStatusType.SUCCESS)
                return result
            elif response == Gtk.ResponseType.APPLY:
                url = "https://kritiksoman.github.io/GIMP-ML-Docs/docs-page.html#item-7-2"
                Gio.app_info_launch_default_for_uri(url, None)
                continue
            else:
                dialog.destroy()
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL, GLib.Error()
                )


class Interpolation(Gimp.PlugIn):
    ## Parameters ##
    __gproperties__ = {
        "file": (
            Gio.File,
            _("Histogram _File"),
            "Histogram export file",
            GObject.ParamFlags.READWRITE,
        ),
        "force_cpu": (
            bool,
            _("Force _CPU"),
            "Force CPU",
            False,
            GObject.ParamFlags.READWRITE,
        ),
    }

    ## GimpPlugIn virtual methods ##
    def do_query_procedures(self):
        self.set_translation_domain(
            "gimp30-python", Gio.file_new_for_path(Gimp.locale_directory())
        )
        return ["interpolation"]

    def do_create_procedure(self, name):
        procedure = None
        if name == "interpolation":
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, run, None
            )
            procedure.set_image_types("*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.DRAWABLES)
            procedure.set_documentation(
                N_("Extracts the monocular depth of the current layer."),
                globals()[
                    "__doc__"
                ],  # This includes the docstring, on the top of the file
                name,
            )
            procedure.set_menu_label(N_("_Interpolation..."))
            procedure.set_attribution("Kritik Soman", "GIMP-ML", "2021")
            procedure.add_menu_path("<Image>/Layer/GIMP-ML/")

            procedure.add_argument_from_property(self, "file")
            procedure.add_argument_from_property(self, "force_cpu")

        return procedure


Gimp.main(Interpolation.__gtype__, sys.argv)
