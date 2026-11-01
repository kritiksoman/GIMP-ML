"""Execute through GIMP's python-fu-eval batch interpreter, never ordinary Python."""

import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import time
import zlib

import gi
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gimp, GimpUi, GLib, Gtk


ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from gimpml import settings


def chunk(name, data):
    return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))


PNG = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 4, 8, 6, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress((b"\x00" + b"\x20\x40\x80\xff" * 8) * 4))
    + chunk(b"IEND", b"")
)

spec = importlib.util.spec_from_file_location(
    "gimpml_native", ROOT / "plug-ins/gimpml-gemini/gimpml-gemini.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
registered = Gimp.get_pdb().lookup_procedure(native.PROCEDURE)
assert registered is not None, "Install the plugin before running this test."
arguments = {argument.name for argument in registered.get_arguments()}
assert {"prompt", "provider", "model", "aspect-ratio", "resolution", "fit-to-canvas"} <= arguments
print("PASS: installed procedure registration and arguments")

images = []
for base_type, layer_type in (
    (Gimp.ImageBaseType.RGB, Gimp.ImageType.RGBA_IMAGE),
    (Gimp.ImageBaseType.GRAY, Gimp.ImageType.GRAYA_IMAGE),
):
    image = Gimp.Image.new(100, 80, base_type)
    images.append(image)
    original = Gimp.Layer.new(image, "Original", 100, 80, layer_type, 73, Gimp.LayerMode.NORMAL)
    original.fill(Gimp.FillType.WHITE)
    image.insert_layer(original, None, 0)
    original_id = original.get_id()
    native.insert_generated_layer(PNG, image, "offline fixture", settings.MODELS[0], True)
    layers = image.get_layers()
    assert len(layers) == 2
    assert layers[1].get_id() == original_id
    assert layers[1].get_name() == "Original" and layers[1].get_opacity() == 73
    assert (layers[0].get_width(), layers[0].get_height()) == (100, 50)
    assert layers[0].get_offsets() == (True, 0, 15)
    assert image.get_width() == 100 and image.get_height() == 80
    print("PASS: new layer, preserved original, aspect-ratio fit", base_type)

native.insert_generated_layer(PNG, images[0], "native size", settings.MODELS[0])
assert (images[0].get_layers()[0].get_width(), images[0].get_layers()[0].get_height()) == (8, 4)
print("PASS: native output dimensions")
new_image = native.insert_generated_layer(PNG, None, "new document", settings.MODELS[0])
images.append(new_image)
assert (new_image.get_width(), new_image.get_height()) == (8, 4)
assert len(new_image.get_layers()) == 1
print("PASS: generation without an open document")

with tempfile.TemporaryDirectory(prefix="gimpml-smoke-") as directory:
    fake = Path(directory) / "fake-python"
    fake.write_text(
        f"#!{ROOT}/.venv/bin/python\nimport json,sys\nfrom pathlib import Path\n"
        "payload=json.load(sys.stdin)\n"
        f"Path(payload['output']).write_bytes({PNG!r})\n"
        "print(json.dumps({'status':'success','width':8,'height':4}))\n")
    fake.chmod(0o755)
    assert native.Worker(ROOT, fake, 10).run({"prompt": "fixture"}) == PNG
    print("PASS: external worker JSON transport")
    actual_runtime = native.runtime
    native.runtime = lambda: (ROOT, fake)
    config = registered.create_config()
    config.set_property("prompt", "offline callback")
    before = len(images[0].get_layers())
    result = native.run(registered, Gimp.RunMode.NONINTERACTIVE, images[0], [], config, None)
    assert result.index(0) == Gimp.PDBStatusType.SUCCESS
    assert len(images[0].get_layers()) == before + 1
    config.set_property("prompt", "")
    result = native.run(registered, Gimp.RunMode.NONINTERACTIVE, images[0], [], config, None)
    assert result.index(0) == Gimp.PDBStatusType.CALLING_ERROR
    assert len(images[0].get_layers()) == before + 1
    native.runtime = actual_runtime
    print("PASS: procedure callback inserts exactly one generated layer")
    try:
        native.Worker(ROOT, ROOT / ".venv/bin/python", 10).run({"prompt": ""})
    except RuntimeError as error:
        assert str(error) == "Enter an image-generation prompt."
    else:
        raise AssertionError("The actual backend must reject an empty prompt without API calls.")
    print("PASS: actual backend interpreter and sanitized validation error")

    GimpUi.init("gimpml-smoke")
    defaults = {**settings.preferences(), "prompt": "offline fixture", "fit_to_canvas": False}
    dialog = native.GenerateDialog(ROOT, fake, settings, defaults)
    GLib.idle_add(lambda: dialog.response(Gtk.ResponseType.OK) or GLib.SOURCE_REMOVE)
    assert dialog.run() == Gtk.ResponseType.OK
    assert dialog.png == PNG
    dialog.destroy()
    print("PASS: native dialog asynchronous generation")

    fake.write_text(
        f"#!{ROOT}/.venv/bin/python\nimport json,sys\njson.load(sys.stdin)\n"
        "print(json.dumps({'status':'error','message':'Insufficient provider credits.'}))\n"
        "sys.exit(1)\n")
    dialog = native.GenerateDialog(ROOT, fake, settings, defaults)
    GLib.idle_add(lambda: dialog.response(Gtk.ResponseType.OK) or GLib.SOURCE_REMOVE)

    def dismiss_error():
        if dialog.worker is not None or not dialog.status.get_text():
            return GLib.SOURCE_CONTINUE
        assert dialog.status.get_text() == "Insufficient provider credits."
        assert dialog.prompt.get_sensitive()
        dialog.response(Gtk.ResponseType.CANCEL)
        return GLib.SOURCE_REMOVE

    GLib.timeout_add(100, dismiss_error)
    assert dialog.run() == Gtk.ResponseType.CANCEL
    assert dialog.png is None
    dialog.destroy()
    print("PASS: actionable provider errors leave inputs editable and insert no layer")

    fake.write_text(
        f"#!{ROOT}/.venv/bin/python\nimport json,sys,time\n"
        "json.load(sys.stdin)\ntime.sleep(30)\n")
    dialog = native.GenerateDialog(ROOT, fake, settings, defaults)
    GLib.idle_add(lambda: dialog.response(Gtk.ResponseType.OK) or GLib.SOURCE_REMOVE)
    GLib.timeout_add(200, lambda: dialog.response(Gtk.ResponseType.CANCEL) or GLib.SOURCE_REMOVE)
    started = time.monotonic()
    assert dialog.run() == Gtk.ResponseType.CANCEL
    assert dialog.png is None and dialog.worker is None
    assert time.monotonic() - started < 5
    dialog.destroy()
    print("PASS: cancellation drains the worker and produces no image")

for image in images:
    image.clean_all()
    if image.get_id() != new_image.get_id():
        image.delete()

print("GIMP-ML OFFLINE SMOKE TEST PASSED")
