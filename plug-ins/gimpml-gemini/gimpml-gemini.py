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
gi.require_version("Gegl", "0.4")
from gi.repository import Gimp, GimpUi, Gegl, Gio, GLib, GObject, Gtk


PROCEDURES = {
    "python-fu-gimpml-generate": ("generate", "_Generate Image..."),
    "python-fu-gimpml-edit-layer": ("layer", "Edit Selected _Layer..."),
    "python-fu-gimpml-edit-canvas": ("canvas", "Edit Visible _Canvas..."),
}
OUTPUT_MODES = (("new-layer", "Add a new layer (preserve originals)"),
                ("replace-layer", "Replace selected layer pixels (undoable)"))
SETTINGS_PROCEDURE = "python-fu-gimpml-settings"


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


def status_markup(message, help_urls):
    markup = GLib.markup_escape_text(message)
    for url in help_urls.values():
        escaped_url = GLib.markup_escape_text(url)
        markup = markup.replace(escaped_url, f'<a href="{escaped_url}">{escaped_url}</a>')
    return markup


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


def selected_layer(image, drawables=None):
    if image is None or not image.is_valid():
        raise RuntimeError("Open an image before editing.")
    if image.get_base_type() == Gimp.ImageBaseType.INDEXED:
        raise RuntimeError("Convert the image to RGB or grayscale before editing.")
    layers = list(drawables) if drawables else image.get_selected_layers()
    if len(layers) != 1 or not isinstance(layers[0], Gimp.Layer) or layers[0].is_group():
        raise RuntimeError("Select exactly one ordinary layer, not a group or layer mask.")
    layer = layers[0]
    if not layer.is_valid() or layer.get_image().get_id() != image.get_id():
        raise RuntimeError("The selected layer is no longer in the target image.")
    return layer


def check_replacement(layer):
    if layer is None or not layer.is_valid():
        raise RuntimeError("The target layer was removed. Select a layer and try again.")
    item = layer
    while item is not None:
        if item.get_lock_content():
            raise RuntimeError("Unlock the selected layer and its parent groups before replacing pixels.")
        item = item.get_parent()
    if layer.get_lock_alpha():
        raise RuntimeError("Unlock the selected layer's alpha channel before replacing pixels.")


def editing_options(image, input_mode, output_mode, drawables=None):
    if input_mode not in ("generate", "layer", "canvas") or output_mode not in dict(OUTPUT_MODES):
        raise RuntimeError("Choose a supported image input and output mode.")
    if input_mode == "generate":
        if output_mode != "new-layer":
            raise RuntimeError("Choose an editing input before replacing a layer.")
        return None
    if image is None or not image.is_valid():
        raise RuntimeError("Open an image before editing.")
    if image.get_base_type() == Gimp.ImageBaseType.INDEXED:
        raise RuntimeError("Convert the image to RGB or grayscale before editing.")
    layer = None
    if input_mode == "layer" or output_mode == "replace-layer":
        layer = selected_layer(image, drawables)
    if output_mode == "replace-layer":
        check_replacement(layer)
    return layer


def unlock_export_layers(snapshot):
    pending = list(snapshot.get_layers())
    while pending:
        layer = pending.pop()
        if (not layer.set_lock_content(False) or not layer.set_lock_position(False)
                or not layer.set_lock_visibility(False) or not layer.set_lock_alpha(False)):
            raise RuntimeError("GIMP could not unlock the temporary image copy for export.")
        if layer.is_group():
            pending.extend(layer.get_children())


def capture_reference(image, input_mode, layer=None):
    from gimpml.settings import MAX_REFERENCE_BYTES, MAX_REFERENCE_PIXELS

    if input_mode == "layer":
        width, height = layer.get_width(), layer.get_height()
        _, x, y = layer.get_offsets()
    else:
        width, height = image.get_width(), image.get_height()
        x, y = 0, 0
    if width * height > MAX_REFERENCE_PIXELS:
        raise RuntimeError("The editing input exceeds 32 million pixels. Reduce the image size.")
    if input_mode == "layer":
        snapshot = Gimp.Image.new(width, height, image.get_base_type())
    else:
        snapshot = image.duplicate()
        if snapshot is None:
            raise RuntimeError("GIMP could not copy the visible canvas.")
    try:
        if input_mode == "layer":
            copied = Gimp.Layer.new(snapshot, "Editing input", width, height,
                                   layer.type_with_alpha(), 100, Gimp.LayerMode.NORMAL)
            if copied is None or not snapshot.insert_layer(copied, None, 0):
                raise RuntimeError("GIMP could not copy the selected layer.")
            rect = Gegl.Rectangle.new(0, 0, width, height)
            buffer = copied.get_buffer()
            layer.get_buffer().copy(rect, Gegl.AbyssPolicy.NONE, buffer, rect)
            buffer.flush()
            if not copied.update(0, 0, width, height):
                raise RuntimeError("GIMP could not prepare the selected layer pixels for upload.")
        else:
            unlock_export_layers(snapshot)
            if snapshot.merge_visible_layers(Gimp.MergeType.CLIP_TO_IMAGE) is None:
                raise RuntimeError("There are no visible layers to edit.")
        with tempfile.TemporaryDirectory(prefix="gimpml-reference-") as directory:
            path = Path(directory) / "input.png"
            exporter = Gimp.get_pdb().lookup_procedure("file-png-export")
            if exporter is None:
                raise RuntimeError("GIMP's PNG exporter is unavailable.")
            config = exporter.create_config()
            config.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
            config.set_property("image", snapshot)
            config.set_property("file", Gio.File.new_for_path(str(path)))
            for name in ("include-exif", "include-iptc", "include-xmp",
                         "include-color-profile", "include-thumbnail", "include-comment",
                         "bkgd", "offs", "phys", "time"):
                config.set_property(name, False)
            result = exporter.run(config)
            if result.index(0) != Gimp.PDBStatusType.SUCCESS or not path.is_file():
                raise RuntimeError("GIMP could not export the editing input as PNG.")
            with path.open("rb") as handle:
                png = handle.read(MAX_REFERENCE_BYTES + 1)
            if len(png) > MAX_REFERENCE_BYTES:
                raise RuntimeError("The editing input exceeds 12 MiB. Reduce the image size.")
            return png, (x, y, width, height)
    finally:
        snapshot.delete()


def apply_edited_image(png, image, prompt, model, bounds, output_mode, target=None):
    if image is None or not image.is_valid():
        raise RuntimeError("The target image was closed. Generate again in an open document.")
    if output_mode == "replace-layer":
        check_replacement(target)
        if target.get_image().get_id() != image.get_id():
            raise RuntimeError("The target layer is no longer in the image.")
    x, y, width, height = bounds
    with tempfile.TemporaryDirectory(prefix="gimpml-edit-") as directory:
        path = Path(directory) / "edited.png"
        path.write_bytes(png)
        result_image = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.File.new_for_path(str(path)))
        if result_image is None:
            raise RuntimeError("GIMP could not load the edited image.")
        try:
            edited = result_image.get_layers()[0]
            scale = min(width / edited.get_width(), height / edited.get_height())
            scaled_width = max(1, round(edited.get_width() * scale))
            scaled_height = max(1, round(edited.get_height() * scale))
            if not edited.scale(scaled_width, scaled_height, False):
                raise RuntimeError("GIMP could not size the edited image.")
            if not edited.resize(width, height, (width - scaled_width) // 2,
                                 (height - scaled_height) // 2) or not edited.set_offsets(0, 0):
                raise RuntimeError("GIMP could not align the edited image.")
            if output_mode == "new-layer":
                layer = Gimp.Layer.new_from_drawable(edited, image)
                if layer is None:
                    raise RuntimeError("GIMP could not create the edited layer.")
                layer.set_name(f"AI edit | {model} | {prompt.strip()[:80]}")
                if not layer.set_offsets(x, y):
                    raise RuntimeError("GIMP could not position the edited layer.")
                image.undo_group_start()
                try:
                    if not image.insert_layer(layer, None, 0):
                        raise RuntimeError("GIMP could not insert the edited layer.")
                finally:
                    image.undo_group_end()
            else:
                _, target_x, target_y = target.get_offsets()
                target_width, target_height = target.get_width(), target.get_height()
                source_rect = Gegl.Rectangle.new(target_x - x, target_y - y,
                                                 target_width, target_height)
                destination_rect = Gegl.Rectangle.new(0, 0, target_width, target_height)
                image.undo_group_start()
                added_alpha, merged = False, False
                try:
                    if not target.has_alpha():
                        if not target.add_alpha():
                            raise RuntimeError("GIMP could not add transparency to the selected layer.")
                        added_alpha = True
                    shadow = target.get_shadow_buffer()
                    edited.get_buffer().copy(source_rect, Gegl.AbyssPolicy.NONE,
                                             shadow, destination_rect)
                    shadow.flush()
                    if not target.merge_shadow(True):
                        raise RuntimeError("GIMP could not replace the selected layer pixels.")
                    merged = True
                    if not target.update(0, 0, target_width, target_height):
                        raise RuntimeError("GIMP could not refresh the edited layer.")
                finally:
                    try:
                        if added_alpha and not merged and not target.remove_alpha():
                            raise RuntimeError("GIMP could not restore the original layer's alpha state.")
                    finally:
                        image.undo_group_end()
        finally:
            result_image.delete()
    Gimp.displays_flush()


def worker_environment(root):
    environment = os.environ.copy()
    # GIMP's bundled Python paths must not leak into the external interpreter.
    for key in list(environment):
        if key.startswith(("PYTHON", "DYLD_", "GI_", "GIMP", "GTK", "GEGL", "BABL")):
            environment.pop(key)
    environment["PYTHONPATH"] = str(root)
    return environment


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

    def run(self, request, reference_png=None):
        with tempfile.TemporaryDirectory(prefix="gimpml-worker-") as directory:
            output = Path(directory) / "generated.png"
            payload = {"request": request, "output": str(output)}
            if reference_png is not None:
                source = Path(directory) / "input.png"
                source.write_bytes(reference_png)
                payload["input"] = str(source)
            with self.lock:
                if self.cancelled.is_set():
                    return None
                self.process = subprocess.Popen(
                    [str(self.python), "-m", "gimpml.worker"], cwd=str(self.root),
                    env=worker_environment(self.root), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True,
                )
            try:
                stdout, _stderr = self.process.communicate(
                    json.dumps(payload),
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
    def __init__(self, root, python, settings, defaults, image=None, drawables=None,
                 input_mode="generate"):
        titles = {"generate": "Generate Image",
                  "layer": "Edit Selected Layer",
                  "canvas": "Edit Visible Canvas"}
        super().__init__(title="GIMP-ML - " + titles[input_mode], modal=True)
        self.set_default_size(560, 430)
        self.root, self.python, self.settings = root, python, settings
        self.timeout = settings.preferences(root)["timeout_seconds"]
        self.worker = None
        self.png = None
        self.request = None
        self.fit_to_canvas = False
        self.image, self.drawables = image, drawables
        self.reference_png = None
        self.bounds = None
        self.target = None
        self.input_mode = input_mode
        self.output_mode = "new-layer"
        self.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        self.add_button("_Generate", Gtk.ResponseType.OK)
        self.connect("response", self.on_response)
        self.connect("delete-event", self.on_delete)
        content = self.get_content_area()
        content.set_border_width(16)
        content.set_spacing(12)
        label = Gtk.Label(label=("Describe the image you want to generate:" if input_mode == "generate"
                                 else "Describe the changes you want to make:"), xalign=0)
        content.pack_start(label, False, False, 0)
        self.prompt = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.prompt.get_buffer().set_text(defaults["prompt"])
        scroll = Gtk.ScrolledWindow()
        scroll.set_min_content_height(120)
        scroll.add(self.prompt)
        content.pack_start(scroll, True, True, 0)
        grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        content.pack_start(grid, False, False, 0)
        self.edit_controls = {}
        if input_mode != "generate":
            control = Gtk.ComboBoxText()
            for key, text in OUTPUT_MODES:
                control.append(key, text)
            control.set_active_id(defaults.get("output_mode", "new-layer"))
            self.edit_controls["output_mode"] = control
            grid.attach(Gtk.Label(label="Edited result", xalign=0), 0, 0, 1, 1)
            grid.attach(control, 1, 0, 1, 1)
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
            offset = 1 if input_mode != "generate" else 0
            grid.attach(Gtk.Label(label=label, xalign=0), 0, row + offset, 1, 1)
            grid.attach(control, 1, row + offset, 1, 1)
        self.controls["model"].connect("changed", self.model_changed)
        self.model_changed()
        self.fit = Gtk.CheckButton(label="Fit image within canvas (preserve aspect ratio)")
        self.fit.set_active(defaults["fit_to_canvas"])
        if input_mode == "generate":
            content.pack_start(self.fit, False, False, 0)
        self.upload_consent = Gtk.CheckButton(
            label="I allow uploading this image to the selected provider for editing.")
        if input_mode != "generate":
            content.pack_start(self.upload_consent, False, False, 0)
        self.disclosure = Gtk.Label(xalign=0, wrap=True)
        content.pack_start(self.disclosure, False, False, 0)
        self.mode_changed()
        self.status = Gtk.Label(xalign=0, wrap=True, selectable=True)
        content.pack_start(self.status, False, False, 0)
        self.show_all()

    def mode_changed(self, *_args):
        editing = self.input_mode != "generate"
        if editing:
            self.edit_controls["output_mode"].set_sensitive(True)
        self.upload_consent.set_active(False)
        self.upload_consent.set_sensitive(editing)
        self.fit.set_sensitive(not editing)
        self.get_widget_for_response(Gtk.ResponseType.OK).set_label(
            "_Apply Edit" if editing else "_Generate")
        disclosure = (
            ("Editing uploads " + ("the selected layer pixels" if self.input_mode == "layer"
                                    else "the visible canvas") + " with your prompt.\n"
             "The result is fitted within the input bounds. Replacement preserves layer masks\n"
             "and properties; canvas replacement is cropped to the selected layer's bounds.\n")
            if editing else "Text-only generation uploads your prompt, not your image.\n")
        self.disclosure.set_text(disclosure +
                                 "API use may incur charges. Cancel may not prevent provider charges.")

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
            self.output_mode = (self.edit_controls["output_mode"].get_active_id()
                                if self.input_mode != "generate" else "new-layer")
            self.target = editing_options(self.image, self.input_mode, self.output_mode, self.drawables)
            self.reference_png, self.bounds = None, None
            if self.input_mode != "generate":
                if not self.upload_consent.get_active():
                    raise RuntimeError("Allow image upload before applying an edit.")
                self.reference_png, self.bounds = capture_reference(
                    self.image, self.input_mode, self.target)
        except (self.settings.GenerationError, RuntimeError, OSError, GLib.Error) as error:
            self.status.set_text(str(error))
            return
        self.fit_to_canvas = self.fit.get_active() if self.input_mode == "generate" else False
        self.worker = Worker(self.root, self.python, self.timeout)
        self.set_response_sensitive(Gtk.ResponseType.OK, False)
        self.prompt.set_sensitive(False)
        for control in self.controls.values():
            control.set_sensitive(False)
        for control in self.edit_controls.values():
            control.set_sensitive(False)
        self.upload_consent.set_sensitive(False)
        self.fit.set_sensitive(False)
        self.status.set_text("Editing image..." if self.reference_png is not None else "Generating image...")
        threading.Thread(target=self.generate, daemon=True).start()

    def generate(self):
        try:
            png = self.worker.run(self.request, self.reference_png)
            error = None
        except (OSError, RuntimeError) as exception:
            png = None
            error = str(exception) if isinstance(exception, RuntimeError) else "Cannot launch the API worker. Re-run the installer."
        GLib.idle_add(self.complete, png, error)

    def complete(self, png, error):
        cancelled = self.worker.cancelled.is_set()
        self.worker = None
        self.reference_png = None
        self.set_response_sensitive(Gtk.ResponseType.CANCEL, True)
        if cancelled:
            self.response(Gtk.ResponseType.CANCEL)
        elif error:
            self.status.set_markup(status_markup(error, self.settings.KEY_HELP_URLS))
            self.set_response_sensitive(Gtk.ResponseType.OK, True)
            self.prompt.set_sensitive(True)
            for control in self.controls.values():
                control.set_sensitive(True)
            self.model_changed()
            self.mode_changed()
        else:
            self.png = png
            self.response(Gtk.ResponseType.OK)
        return GLib.SOURCE_REMOVE


def credential_request(root, python, payload):
    try:
        process = subprocess.run(
            [str(python), "-m", "gimpml.credentials"], cwd=str(root),
            env=worker_environment(root), input=json.dumps(payload), capture_output=True,
            text=True, timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Keychain access timed out. Some changes may have been saved. "
                           "Unlock Keychain and reopen Settings to check.") from error
    except OSError as error:
        raise RuntimeError("Cannot launch the settings worker. Run scripts/install.py again.") from error
    try:
        result = json.loads(process.stdout)
    except ValueError as error:
        raise RuntimeError("The settings worker failed. Run scripts/install.py again.") from error
    if not isinstance(result, dict):
        raise RuntimeError("The settings worker returned an invalid result.")
    if result.get("status") != "success" or process.returncode != 0:
        message = result.get("message")
        raise RuntimeError(message if isinstance(message, str) else "The settings worker failed.")
    saved = result.get("saved")
    if (not isinstance(saved, dict) or set(saved) != {"openrouter", "google"}
            or any(type(value) is not bool for value in saved.values())):
        raise RuntimeError("The settings worker returned invalid credential status.")
    return saved


class SettingsDialog(Gtk.Dialog):
    def __init__(self, root, python):
        sys.path.insert(0, str(root))
        from gimpml.settings import KEY_HELP_URLS

        super().__init__(title="GIMP-ML - Settings", modal=True)
        self.root, self.python = root, python
        self.busy = False
        self.saved = False
        self.entries, self.remove, self.key_status = {}, {}, {}
        self.key_help = {}
        self.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        self.add_button("_Save", Gtk.ResponseType.OK)
        self.connect("response", self.on_response)
        self.connect("delete-event", self.on_delete)
        content = self.get_content_area()
        content.set_border_width(16)
        content.set_spacing(12)
        content.pack_start(Gtk.Label(
            label="API keys are stored in macOS Keychain, not in the project.\n"
                  "Leave a field blank to keep its saved key. Check Remove to delete it.\n"
                  "Worker environment variables override Keychain.\n"
                  "Removing a saved key does not remove environment credentials.\n"
                  "macOS may ask you to unlock Keychain or allow access.",
            xalign=0, wrap=True), False, False, 0)
        for provider, label in (("openrouter", "OpenRouter API key"),
                                ("google", "Google API key")):
            content.pack_start(Gtk.Label(label=label, xalign=0), False, False, 0)
            entry = Gtk.Entry(visibility=False)
            entry.set_max_length(4096)
            entry.set_placeholder_text("Enter a new key (blank keeps the saved key)")
            content.pack_start(entry, False, False, 0)
            self.entries[provider] = entry
            help_link = Gtk.LinkButton.new_with_label(
                KEY_HELP_URLS[provider], "How to create your " + label)
            help_link.set_halign(Gtk.Align.START)
            content.pack_start(help_link, False, False, 0)
            self.key_help[provider] = help_link
            status = Gtk.Label(label="Checking saved key...", xalign=0)
            content.pack_start(status, False, False, 0)
            self.key_status[provider] = status
            remove = Gtk.CheckButton(label="Remove saved " + label)
            content.pack_start(remove, False, False, 0)
            self.remove[provider] = remove
        self.status = Gtk.Label(xalign=0, wrap=True)
        content.pack_start(self.status, False, False, 0)
        self.show_all()
        self.start({"operation": "status"})

    def start(self, payload):
        self.busy = True
        for response in (Gtk.ResponseType.OK, Gtk.ResponseType.CANCEL):
            self.set_response_sensitive(response, False)
        for widget in (*self.entries.values(), *self.remove.values()):
            widget.set_sensitive(False)
        self.status.set_text("Accessing macOS Keychain...")
        threading.Thread(target=self.execute, args=(payload,), daemon=True).start()

    def execute(self, payload):
        try:
            result = credential_request(self.root, self.python, payload)
            error = None
        except RuntimeError as exception:
            result, error = None, str(exception)
        GLib.idle_add(self.complete, result, error, payload["operation"] == "save")

    def complete(self, result, error, saving):
        self.busy = False
        for response in (Gtk.ResponseType.OK, Gtk.ResponseType.CANCEL):
            self.set_response_sensitive(response, True)
        for widget in (*self.entries.values(), *self.remove.values()):
            widget.set_sensitive(True)
        if error:
            for status in self.key_status.values():
                status.set_text("Saved-key status unavailable.")
            self.status.set_text(error)
        else:
            for provider, exists in result.items():
                self.key_status[provider].set_text("Key saved in Keychain." if exists else "No saved key.")
            self.status.set_text("Keys are not displayed or checked against the provider.")
            if saving:
                self.saved = True
                self.response(Gtk.ResponseType.OK)
        return GLib.SOURCE_REMOVE

    def on_delete(self, *_args):
        return self.busy

    def on_response(self, _dialog, response):
        if self.saved:
            return
        if self.busy:
            self.stop_emission_by_name("response")
            return
        if response != Gtk.ResponseType.OK:
            return
        self.stop_emission_by_name("response")
        keys = {provider: entry.get_text().strip() for provider, entry in self.entries.items()
                if entry.get_text().strip()}
        clear = [provider for provider, control in self.remove.items() if control.get_active()]
        if set(keys) & set(clear):
            self.status.set_text("Choose either a new key or removal for each provider, not both.")
            return
        for entry in self.entries.values():
            entry.set_text("")
        self.start({"operation": "save", "keys": keys, "clear": clear})


def run_settings(procedure, run_mode, _image, _drawables, _config, _data):
    if run_mode != Gimp.RunMode.INTERACTIVE:
        Gimp.message("Open GIMP-ML Settings interactively to manage API keys.")
        return procedure.new_return_values(Gimp.PDBStatusType.CALLING_ERROR, GLib.Error())
    try:
        root, python = runtime()
        GimpUi.init(SETTINGS_PROCEDURE)
        dialog = SettingsDialog(root, python)
        response = dialog.run()
        dialog.destroy()
        status = Gimp.PDBStatusType.SUCCESS if response == Gtk.ResponseType.OK else Gimp.PDBStatusType.CANCEL
        return procedure.new_return_values(status, GLib.Error())
    except (RuntimeError, OSError, GLib.Error) as error:
        Gimp.message(str(error))
        return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, GLib.Error(str(error)))


def run(procedure, run_mode, image, drawables, config, _data):
    try:
        root, python = runtime()
        sys.path.insert(0, str(root))
        from gimpml import settings
        defaults = settings.preferences(root)
        request = {name: config.get_property(name) for name in
                   ("prompt", "provider", "model", "aspect_ratio", "resolution")}
        input_mode = PROCEDURES[procedure.get_name()][0]
        editing = input_mode != "generate"
        fit = config.get_property("fit-to-canvas") if not editing else False
        output_mode = config.get_property("output-mode") if editing else "new-layer"
        allow_upload = config.get_property("allow-image-upload") if editing else False
        bounds, target = None, None
        if run_mode == Gimp.RunMode.INTERACTIVE:
            GimpUi.init(procedure.get_name())
            request.update({key: defaults[key] for key in request if key != "prompt"})
            dialog = GenerateDialog(root, python, settings, {
                **request, "fit_to_canvas": fit,
                "output_mode": output_mode,
            }, image, drawables, input_mode)
            response = dialog.run()
            png, request, fit = dialog.png, dialog.request, dialog.fit_to_canvas
            input_mode, output_mode = dialog.input_mode, dialog.output_mode
            bounds, target = dialog.bounds, dialog.target
            allow_upload = dialog.upload_consent.get_active()
            dialog.destroy()
            if response != Gtk.ResponseType.OK:
                return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())
        else:
            settings.GenerationRequest(**request).validate()
            target = editing_options(image, input_mode, output_mode, drawables)
            reference = None
            if input_mode != "generate":
                if not allow_upload:
                    raise RuntimeError("Set allow-image-upload to true before editing through the procedure.")
                reference, bounds = capture_reference(image, input_mode, target)
            png = Worker(root, python, defaults["timeout_seconds"]).run(request, reference)
        if input_mode == "generate":
            insert_generated_layer(png, image, request["prompt"], request["model"], fit)
        else:
            apply_edited_image(png, image, request["prompt"], request["model"],
                               bounds, output_mode, target)
        for name, value in request.items():
            config.set_property(name, value)
        if editing:
            config.set_property("output-mode", output_mode)
            config.set_property("allow-image-upload", allow_upload)
        else:
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
        return list(PROCEDURES) + ([SETTINGS_PROCEDURE] if sys.platform == "darwin" else [])

    def do_create_procedure(self, name):
        if name == SETTINGS_PROCEDURE:
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, run_settings, None)
            procedure.set_image_types("*")
            procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.ALWAYS)
            procedure.set_menu_label("_Settings...")
            procedure.add_menu_path("<Image>/Filters/GIMP-ML")
            procedure.set_documentation("Manage API keys in macOS Keychain.",
                                        "Masked key fields; credentials are not saved in procedure arguments.",
                                        name)
            procedure.set_attribution("Kritik Soman", "Kritik Soman", "2026")
            return procedure
        if name not in PROCEDURES:
            return None
        input_mode, label = PROCEDURES[name]
        procedure = Gimp.ImageProcedure.new(self, name, Gimp.PDBProcType.PLUGIN, run, None)
        procedure.set_image_types("RGB*, GRAY*")
        procedure.set_sensitivity_mask(Gimp.ProcedureSensitivityMask.ALWAYS if input_mode == "generate"
                                       else Gimp.ProcedureSensitivityMask.DRAWABLE)
        procedure.set_menu_label(label)
        procedure.add_menu_path("<Image>/Filters/GIMP-ML")
        procedure.set_documentation(
            label.replace("_", "").rstrip("."),
            ("Sends a text prompt only and adds the generated image as a new layer."
             if input_mode == "generate" else
             "Uploads " + ("the selected layer" if input_mode == "layer" else "the visible canvas")
             + " with a text prompt. Adds a new layer or replaces selected-layer pixels with undo."),
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
        if input_mode == "generate":
            procedure.add_boolean_argument("fit-to-canvas", "Fit to canvas",
                                           "Scale to fit without stretching", False, flags)
        else:
            procedure.add_string_argument("output-mode", "Output (new-layer, replace-layer)",
                                          "Where to put the edited result", "new-layer", flags)
            procedure.add_boolean_argument("allow-image-upload", "Allow image upload",
                                           "Consent to sending image pixels to the provider for editing",
                                           False, flags)
        return procedure


if __name__ == "__main__":
    Gimp.main(GeminiPlugin.__gtype__, sys.argv)
