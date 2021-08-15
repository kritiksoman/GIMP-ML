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


Performs RGB to greyscale conversion of the current layer with optional color mask layer.
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


def coloring(procedure, image, n_drawables, drawables, force_cpu, progress_bar, config_path_output):
    # Save inference parameters and layers
    weight_path = config_path_output["weight_path"]
    python_path = config_path_output["python_path"]
    plugin_path = config_path_output["plugin_path"]

    Gimp.context_push()
    image.undo_group_start()

    for index, drawable in enumerate(drawables):
        save_image(image, [drawable], os.path.join(weight_path, "..", "cache" + str(index) + ".png"))

    with open(os.path.join(weight_path, "..", "gimp_ml_run.pkl"), "wb") as file:
        pickle.dump(
            {
                "force_cpu": bool(force_cpu),
                "n_drawables": n_drawables,
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
        result = Gimp.file_load(
            Gimp.RunMode.NONINTERACTIVE,
            Gio.file_new_for_path(os.path.join(weight_path, "..", "cache.png")),
        )
        result_layer = result.get_active_layer()
        copy = Gimp.Layer.new_from_drawable(result_layer, image)
        copy.set_name("Coloring")
        copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)  # DIFFERENCE_LEGACY
        image.insert_layer(copy, None, -1)

        # Remove temporary layers that were saved
        my_dir = os.path.join(weight_path, "..")
        for f_name in os.listdir(my_dir):
            if f_name.startswith("cache"):
                os.remove(os.path.join(my_dir, f_name))

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
    force_cpu = args.index(1)

    if run_mode == Gimp.RunMode.INTERACTIVE:
        # Get all paths
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "..", "tools"
        )
        with open(os.path.join(config_path, "gimp_ml_config.pkl"), "rb") as file:
            config_path_output = pickle.load(file)
        python_path = config_path_output["python_path"]
        config_path_output["plugin_path"] = os.path.join(config_path, "coloring.py")

        config = procedure.create_config()
        config.set_property("force_cpu", force_cpu)
        config.begin_run(image, run_mode, args)

        GimpUi.init("coloring.py")
        use_header_bar = Gtk.Settings.get_default().get_property(
            "gtk-dialogs-use-header"
        )

        # Check number of selected layers
        if n_drawables > 2:
            show_dialog(
                "Please select only greyscale layer and color mask layer.",
                "Error !",
                "error",
                image_paths
            )
            return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())
        elif n_drawables == 1:
            n_drawables_text = _("|  No Color Mask Selected")
        elif n_drawables == 2:
            n_drawables_text = _("|  Color Mask Selected")

        # Create UI
        dialog = GimpUi.Dialog(use_header_bar=use_header_bar, title=_("Coloring..."))
        dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("_Help", Gtk.ResponseType.APPLY)
        dialog.add_button("_Color Palette", Gtk.ResponseType.YES)
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

        # Show n_drawables text
        label = Gtk.Label(label=n_drawables_text)
        grid.attach(label, 1, 0, 1, 1)
        # vbox.pack_start(label, False, False, 1)
        label.show()

        progress_bar = Gtk.ProgressBar()
        vbox.add(progress_bar)
        progress_bar.show()

        # Force CPU Button
        spin = GimpUi.prop_check_button_new(config, "force_cpu", _("Force _CPU"))
        spin.set_tooltip_text(
            _(
                "If checked, CPU is used for model inference."
                " Otherwise, GPU will be used if available."
            )
        )
        grid.attach(spin, 0, 0, 1, 1)
        spin.show()

        # Wait for user to click
        dialog.show()
        while True:
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                force_cpu = config.get_property("force_cpu")
                result = coloring(
                    procedure,
                    image,
                    n_drawables,
                    layer,
                    force_cpu,
                    progress_bar,
                    config_path_output,
                )
                # If the execution was successful, save parameters so they will be restored next time we show dialog.
                if result.index(0) == Gimp.PDBStatusType.SUCCESS and config is not None:
                    config.end_run(Gimp.PDBStatusType.SUCCESS)
                return result
            elif response == Gtk.ResponseType.APPLY:
                url = "https://kritiksoman.github.io/GIMP-ML-Docs/docs-page.html#item-7-13"
                Gio.app_info_launch_default_for_uri(url, None)
                continue
            elif response == Gtk.ResponseType.YES:
                image_new = Gimp.Image.new(1200, 675, 0)  # 0 for RGB
                display = Gimp.Display.new(image_new)
                result = Gimp.file_load(
                    Gimp.RunMode.NONINTERACTIVE,
                    Gio.file_new_for_path(image_paths["colorpalette"]),
                )
                result_layer = result.get_active_layer()
                copy = Gimp.Layer.new_from_drawable(result_layer, image_new)
                copy.set_name("Color Palette")
                copy.set_mode(Gimp.LayerMode.NORMAL_LEGACY)  # DIFFERENCE_LEGACY
                image_new.insert_layer(copy, None, -1)
                Gimp.displays_flush()
                continue
            else:
                dialog.destroy()
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CANCEL, GLib.Error()
                )


class Coloring(Gimp.PlugIn):
    # Parameters #
    __gproperties__ = {
        "force_cpu": (
            bool,
            _("Force _CPU"),
            "Force CPU",
            False,
            GObject.ParamFlags.READWRITE,
        ),
    }

    # GimpPlugIn virtual methods #
    def do_query_procedures(self):
        self.set_translation_domain(
            "gimp30-python", Gio.file_new_for_path(Gimp.locale_directory())
        )
        return ["coloring"]

    def do_create_procedure(self, name):
        procedure = None
        if name == "coloring":
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, run, None
            )
            procedure.set_image_types("*")
            procedure.set_sensitivity_mask(
                Gimp.ProcedureSensitivityMask.DRAWABLE
                | Gimp.ProcedureSensitivityMask.DRAWABLES
            )
            procedure.set_documentation(
                N_(
                    "Performs RGB to greyscale conversion of the current layer with optional color mask layer."
                ),
                globals()[
                    "__doc__"
                ],  # This includes the docstring, on the top of the file
                name,
            )
            procedure.set_menu_label(N_("_Coloring..."))
            procedure.set_attribution("Kritik Soman", "GIMP-ML", "2021")
            procedure.add_menu_path("<Image>/Layer/GIMP-ML/")
            procedure.add_argument_from_property(self, "force_cpu")
        return procedure


Gimp.main(Coloring.__gtype__, sys.argv)
