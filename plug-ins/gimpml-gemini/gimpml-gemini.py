#!/usr/bin/env python3
"""Native GIMP 3 dialog with an isolated, cancellable API worker."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import gi

gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gimp, GimpUi, Gio, GLib, GObject, Gtk


PROCEDURE = "python-fu-gimpml-gemini"


def runtime():
    try:
        config = json.loads(Path(__file__).with_name("runtime.json").read_text())
        root = Path(config["root"])
        python = Path(config["python"])
        if not root.is_dir() or not python.is_file():
            raise ValueError("missing runtime")
        return root, python
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("GIMP-ML installation is missing. Run scripts/install.py again.") from error


def insert_generated_layer(png, image, prompt, model, fit_to_canvas=False):
    with tempfile.TemporaryDirectory(prefix="gimpml-layer-") as directory:
        path = Path(directory) / "generated.png"
        path.write_bytes(png)
        file = Gio.File.new_for_path(str(path))
        if image is None:
            image = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, file)
            if image is None:
                raise RuntimeError("GIMP could not load the generated image.")
            image.get_layers()[0].set_name(f"AI | {model} | {prompt.strip()[:80]}")
            try:
                Gimp.Display.new(image)
            except TypeError as error:
                image.delete()
                raise RuntimeError("GIMP cannot open a display. Use graphical GIMP to create a new document.") from error
        else:
            if not image.is_valid():
                raise RuntimeError("The target image was closed. Generate again in an open document.")
            layer = Gimp.file_load_layer(Gimp.RunMode.NONINTERACTIVE, image, file)
            if layer is None:
                raise RuntimeError("GIMP could not load the generated layer.")
            layer.set_name(f"AI | {model} | {prompt.strip()[:80]}")
            image.undo_group_start()
            inserted = False
            try:
                if not image.insert_layer(layer, None, 0):
                    raise RuntimeError("GIMP could not insert the generated layer.")
                inserted = True
                if fit_to_canvas:
                    width, height = image.get_width(), image.get_height()
                    scale = min(width / layer.get_width(), height / layer.get_height())
                    if not layer.scale(max(1, round(layer.get_width() * scale)),
                                       max(1, round(layer.get_height() * scale)), False):
                        raise RuntimeError("GIMP could not scale the generated layer.")
                    if not layer.set_offsets((width - layer.get_width()) // 2,
                                             (height - layer.get_height()) // 2):
                        raise RuntimeError("GIMP could not position the generated layer.")
            except (RuntimeError, GLib.Error):
                if inserted:
                    image.remove_layer(layer)
                raise
            finally:
                image.undo_group_end()
    Gimp.displays_flush()
    return image


class Worker:
    def __init__(self, root, python, timeout):
        self.root = root
        self.python = python
        self.timeout = timeout
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.process = None

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()

    def run(self, request):
        with tempfile.TemporaryDirectory(prefix="gimpml-worker-") as directory:
            output = Path(directory) / "generated.png"
            environment = os.environ.copy()
            # GIMP's bundled Python paths must not leak into the external interpreter.
            for key in list(environment):
                if key.startswith(("PYTHON", "DYLD_", "GI_", "GIMP", "GTK", "GEGL", "BABL")):
                    environment.pop(key)
            environment["PYTHONPATH"] = str(self.root)
            with self.lock:
                if self.cancelled.is_set():
                    return None
                self.process = subprocess.Popen(
                    [str(self.python), "-m", "gimpml.worker"], cwd=str(self.root),
                    env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True,
                )
            try:
                stdout, _stderr = self.process.communicate(
                    json.dumps({"request": request, "output": str(output)}),
                    timeout=self.timeout + 30,
                )
            except subprocess.TimeoutExpired as error:
                self.process.kill()
                self.process.communicate()
                raise RuntimeError("Generation timed out. The provider may still charge for the request.") from error
            if self.cancelled.is_set():
                return None
            try:
                result = json.loads(stdout)
                if not isinstance(result, dict):
                    raise ValueError("invalid worker result")
            except ValueError as error:
                raise RuntimeError("The API worker failed. Re-run scripts/install.py to check dependencies.") from error
            if result.get("status") != "success" or self.process.returncode != 0:
                raise RuntimeError(result.get("message", "The API worker returned an unexpected error."))
            if not output.is_file():
                raise RuntimeError("The API worker returned no image file.")
            return output.read_bytes()


class GenerateDialog(Gtk.Dialog):
    def __init__(self, root, python, settings, defaults):
        super().__init__(title="GIMP-ML - Gemini Image Generation", modal=True)
        self.set_default_size(560, 430)
        self.root, self.python, self.settings = root, python, settings
        self.timeout = settings.preferences(root)["timeout_seconds"]
        self.worker = None
        self.png = None
        self.request = None
        self.fit_to_canvas = False
        self.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        self.add_button("_Generate", Gtk.ResponseType.OK)
        self.connect("response", self.on_response)
        self.connect("delete-event", self.on_delete)
        content = self.get_content_area()
        content.set_border_width(16)
        content.set_spacing(12)
        label = Gtk.Label(label="Describe the image you want to generate:", xalign=0)
        content.pack_start(label, False, False, 0)
        self.prompt = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.prompt.get_buffer().set_text(defaults["prompt"])
        scroll = Gtk.ScrolledWindow()
        scroll.set_min_content_height(120)
        scroll.add(self.prompt)
        content.pack_start(scroll, True, True, 0)
        grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        content.pack_start(grid, False, False, 0)
        self.controls = {}
        for row, (name, label, choices) in enumerate((
            ("provider", "Provider", settings.PROVIDERS),
            ("model", "Gemini image model", settings.MODELS),
            ("aspect_ratio", "Aspect ratio", settings.RATIOS),
            ("resolution", "Resolution", settings.RESOLUTIONS),
        )):
            control = Gtk.ComboBoxText()
            for choice in choices:
                control.append(choice, choice)
            control.set_active_id(defaults[name])
            self.controls[name] = control
            grid.attach(Gtk.Label(label=label, xalign=0), 0, row, 1, 1)
            grid.attach(control, 1, row, 1, 1)
        self.controls["model"].connect("changed", self.model_changed)
        self.model_changed()
        self.fit = Gtk.CheckButton(label="Fit image within canvas (preserve aspect ratio)")
        self.fit.set_active(defaults["fit_to_canvas"])
        content.pack_start(self.fit, False, False, 0)
        content.pack_start(Gtk.Label(
            label="Your prompt is sent to the selected provider. Generation may incur charges.\n"
                  "No canvas image is uploaded. Cancel may not prevent provider charges.",
            xalign=0, wrap=True), False, False, 0)
        self.status = Gtk.Label(xalign=0, wrap=True, selectable=True)
        content.pack_start(self.status, False, False, 0)
        self.show_all()

    def model_changed(self, *_args):
        supports_size = self.controls["model"].get_active_id() != self.settings.MODELS[0]
        self.controls["resolution"].set_sensitive(supports_size)
        if not supports_size:
            self.controls["resolution"].set_active_id("1K")

    def on_delete(self, *_args):
        if self.worker is not None:
            self.cancel()
            return True
        return False

    def cancel(self):
        self.status.set_text("Cancelling... The provider may still charge for this request.")
        self.set_response_sensitive(Gtk.ResponseType.CANCEL, False)
        self.worker.cancel()

    def on_response(self, _dialog, response):
        if self.png is not None:
            return
        if self.worker is not None:
            self.stop_emission_by_name("response")
            if response in (Gtk.ResponseType.CANCEL, Gtk.ResponseType.DELETE_EVENT):
                self.cancel()
            return
        if response != Gtk.ResponseType.OK:
            return
        self.stop_emission_by_name("response")
        buffer = self.prompt.get_buffer()
        self.request = {name: control.get_active_id() for name, control in self.controls.items()}
        self.request["prompt"] = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), False)
        try:
            self.settings.GenerationRequest(**self.request).validate()
        except self.settings.GenerationError as error:
            self.status.set_text(str(error))
            return
        self.fit_to_canvas = self.fit.get_active()
        self.worker = Worker(self.root, self.python, self.timeout)
        self.set_response_sensitive(Gtk.ResponseType.OK, False)
        self.prompt.set_sensitive(False)
        for control in self.controls.values():
            control.set_sensitive(False)
        self.fit.set_sensitive(False)
        self.status.set_text("Generating image...")
        threading.Thread(target=self.generate, daemon=True).start()

    def generate(self):
        try:
            png = self.worker.run(self.request)
            error = None
        except (OSError, RuntimeError) as exception:
            png = None
            error = str(exception) if isinstance(exception, RuntimeError) else "Cannot launch the API worker. Re-run the installer."
        GLib.idle_add(self.complete, png, error)

    def complete(self, png, error):
        cancelled = self.worker.cancelled.is_set()
        self.worker = None
        self.set_response_sensitive(Gtk.ResponseType.CANCEL, True)
        if cancelled:
            self.response(Gtk.ResponseType.CANCEL)
        elif error:
            self.status.set_text(error)
            self.set_response_sensitive(Gtk.ResponseType.OK, True)
            self.prompt.set_sensitive(True)
            for control in self.controls.values():
                control.set_sensitive(True)
            self.model_changed()
            self.fit.set_sensitive(True)
        else:
            self.png = png
            self.response(Gtk.ResponseType.OK)
        return GLib.SOURCE_REMOVE


def run(procedure, run_mode, image, drawables, config, _data):
    try:
        root, python = runtime()
        sys.path.insert(0, str(root))
        from gimpml import settings
        defaults = settings.preferences(root)
        request = {name: config.get_property(name) for name in
                   ("prompt", "provider", "model", "aspect_ratio", "resolution")}
        fit = config.get_property("fit-to-canvas")
        if run_mode == Gimp.RunMode.INTERACTIVE:
            GimpUi.init(PROCEDURE)
            request.update({key: defaults[key] for key in request if key != "prompt"})
            dialog = GenerateDialog(root, python, settings, {**request, "fit_to_canvas": fit})
            response = dialog.run()
            png, request, fit = dialog.png, dialog.request, dialog.fit_to_canvas
            dialog.destroy()
            if response != Gtk.ResponseType.OK:
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())
        else:
            settings.GenerationRequest(**request).validate()
            png = Worker(root, python, defaults["timeout_seconds"]).run(request)
        insert_generated_layer(png, image, request["prompt"], request["model"], fit)
        for name, value in request.items():
            config.set_property(name, value)
        config.set_property("fit-to-canvas", fit)
        return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())
    except (RuntimeError, OSError, ValueError, GLib.Error) as error:
        Gimp.message(str(error))
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error(str(error)))
    except settings.GenerationError as error:
        Gimp.message(str(error))
        return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, GLib.Error(str(error)))


class GeminiPlugin(Gimp.PlugIn):
    def do_set_i18n(self, _name):
        return False, None, None

    def do_query_procedures(self):
        return [PROCEDURE]

    def do_create_procedure(self, name):
        procedure = Gimp.ImageProcedure.new(self, name, Gimp.PDBProcType.PLUGIN, run, None)
        procedure.set_image_types("RGB*, GRAY*")
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.ALWAYS)
        procedure.set_menu_label("Generate Image with _Gemini...")
        procedure.add_menu_path("<Image>/Filters/GIMP-ML")
        procedure.set_documentation(
            "Generate an image with Gemini and add it as a new layer.",
            "Sends a text prompt to OpenRouter or Google. Existing layers remain unchanged.",
            name,
        )
        procedure.set_attribution("Kritik Soman", "Kritik Soman", "2026")
        flags = GObject.ParamFlags.READWRITE
        for key, label, default in (
            ("prompt", "Prompt", ""),
            ("provider", "Provider", "openrouter"),
            ("model", "Gemini image model", "gemini-2.5-flash-image"),
            ("aspect_ratio", "Aspect ratio", "1:1"),
            ("resolution", "Resolution", "1K"),
        ):
            procedure.add_string_argument(key, label, label, default, flags)
        procedure.add_boolean_argument("fit-to-canvas", "Fit to canvas",
                                       "Scale to fit without stretching", False, flags)
        return procedure


if __name__ == "__main__":
    Gimp.main(GeminiPlugin.__gtype__, sys.argv)
