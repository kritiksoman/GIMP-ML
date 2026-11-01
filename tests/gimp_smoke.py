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
from gi.repository import Gimp, GimpUi, Gegl, GLib, Gtk


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
registered_procedures = {}
assert Gimp.get_pdb().lookup_procedure("python-fu-gimpml-gemini") is None
for name, (mode, label) in native.PROCEDURES.items():
    procedure = Gimp.get_pdb().lookup_procedure(name)
    assert procedure is not None, "Install the plugin before running this test."
    assert procedure.get_menu_paths() == ["<Image>/Filters/GIMP-ML"]
    assert procedure.get_menu_label() == label
    arguments = {argument.name for argument in procedure.get_arguments()}
    assert {"prompt", "provider", "model", "aspect-ratio", "resolution"} <= arguments
    assert "input-mode" not in arguments
    if mode == "generate":
        assert "fit-to-canvas" in arguments
        assert not {"output-mode", "allow-image-upload"} & arguments
    else:
        assert {"output-mode", "allow-image-upload"} <= arguments
        assert "fit-to-canvas" not in arguments
    registered_procedures[mode] = procedure
registered = registered_procedures["generate"]
print("PASS: three separate procedures, fixed inputs, menu labels, and arguments")
settings_procedure = Gimp.get_pdb().lookup_procedure(native.SETTINGS_PROCEDURE)
assert settings_procedure is not None
assert settings_procedure.get_menu_label() == "_Settings..."
assert settings_procedure.get_menu_paths() == ["<Image>/Filters/GIMP-ML"]
assert not {"openrouter-key", "google-key", "api-key"} & {
    argument.name for argument in settings_procedure.get_arguments()
}
print("PASS: settings registration without secret procedure arguments")

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


def pixels(layer):
    return bytes(layer.get_buffer().get(
        Gegl.Rectangle.new(0, 0, layer.get_width(), layer.get_height()),
        1.0, "R'G'B'A u8", Gegl.AbyssPolicy.NONE))


def png_pixels(png):
    with tempfile.TemporaryDirectory(prefix="gimpml-test-png-") as directory:
        path = Path(directory) / "fixture.png"
        path.write_bytes(png)
        loaded = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, native.Gio.File.new_for_path(str(path)))
        try:
            return pixels(loaded.get_layers()[0])
        finally:
            loaded.delete()


for base_type, layer_type in (
    (Gimp.ImageBaseType.RGB, Gimp.ImageType.RGBA_IMAGE),
    (Gimp.ImageBaseType.GRAY, Gimp.ImageType.GRAYA_IMAGE),
):
    edit_image = Gimp.Image.new(40, 30, base_type)
    images.append(edit_image)
    target = Gimp.Layer.new(edit_image, "Preserved name", 20, 10, layer_type,
                           73, Gimp.LayerMode.NORMAL)
    target.fill(Gimp.FillType.WHITE)
    edit_image.insert_layer(target, None, 0)
    target.set_offsets(7, 9)
    mask = target.create_mask(Gimp.AddMaskType.BLACK)
    assert target.add_mask(mask)
    assert edit_image.set_selected_layers([target])
    before = pixels(target)
    mask_before = pixels(mask)
    target_id = target.get_id()
    assert native.editing_options(edit_image, "layer", "new-layer") == target
    reference, bounds = native.capture_reference(edit_image, "layer", target)
    assert reference.startswith(b"\x89PNG")
    assert struct.unpack(">II", reference[16:24]) == (20, 10)
    assert bounds == (7, 9, 20, 10)
    assert png_pixels(reference) == b"\xff\xff\xff\xff" * 200
    assert pixels(target) == before and pixels(mask) == mask_before
    assert target.get_offsets() == (True, 7, 9) and target.get_opacity() == 73
    target.set_lock_content(True)
    target.set_lock_position(True)
    target.set_lock_visibility(True)
    target.set_lock_alpha(True)
    assert native.capture_reference(edit_image, "layer", target)[1] == bounds
    native.capture_reference(edit_image, "canvas")
    assert target.get_lock_content() and target.get_lock_position()
    assert target.get_lock_visibility() and target.get_lock_alpha()
    target.set_lock_content(False)
    target.set_lock_position(False)
    target.set_lock_visibility(False)
    target.set_lock_alpha(False)
    native.apply_edited_image(PNG, edit_image, "edit fixture", settings.MODELS[0],
                              bounds, "new-layer", target)
    result = edit_image.get_layers()[0]
    assert len(edit_image.get_layers()) == 2
    assert (result.get_width(), result.get_height()) == (20, 10)
    assert result.get_offsets() == (True, 7, 9)
    assert pixels(target) == before
    assert edit_image.set_selected_layers([target])
    native.apply_edited_image(PNG, edit_image, "edit fixture", settings.MODELS[0],
                              bounds, "replace-layer", target)
    assert len(edit_image.get_layers()) == 2 and target.get_id() == target_id
    assert pixels(target) != before
    assert pixels(mask) == mask_before
    assert target.get_name() == "Preserved name" and target.get_opacity() == 73
    assert target.get_offsets() == (True, 7, 9)
    assert target.get_mode() == Gimp.LayerMode.NORMAL
    assert target.get_mask().get_id() == mask.get_id()
    assert edit_image.undo_is_enabled()
    edit_image.select_rectangle(Gimp.ChannelOps.REPLACE, 7, 9, 5, 5)
    target.fill(Gimp.FillType.WHITE)
    selection_before = pixels(target)
    native.apply_edited_image(PNG, edit_image, "selected edit", settings.MODELS[0],
                              bounds, "replace-layer", target)
    selection_after = pixels(target)
    assert selection_after[:4] != selection_before[:4]
    assert selection_after[10 * 4:11 * 4] == selection_before[10 * 4:11 * 4]
    assert Gimp.Selection.bounds(edit_image) == (True, True, 7, 9, 12, 14)
    Gimp.Selection.none(edit_image)
    target.set_lock_content(True)
    try:
        native.editing_options(edit_image, "layer", "replace-layer")
    except RuntimeError as error:
        assert "Unlock" in str(error)
    else:
        raise AssertionError("Locked layers must not be replaced.")
    target.set_lock_content(False)
    target.set_lock_alpha(True)
    try:
        native.check_replacement(target)
    except RuntimeError as error:
        assert "alpha" in str(error)
    else:
        raise AssertionError("Alpha-locked layers must not be replaced.")
    target.set_lock_alpha(False)
    result.set_visible(False)
    target.set_visible(False)
    try:
        native.capture_reference(edit_image, "canvas")
    except RuntimeError as error:
        assert "visible" in str(error)
    else:
        raise AssertionError("An entirely hidden canvas must not be uploaded.")
    target.set_visible(True)
    canvas, canvas_bounds = native.capture_reference(edit_image, "canvas")
    assert struct.unpack(">II", canvas[16:24]) == (40, 30)
    assert canvas_bounds == (0, 0, 40, 30)
    assert not any(png_pixels(canvas)[3::4])
    native.apply_edited_image(PNG, edit_image, "canvas edit", settings.MODELS[0],
                              canvas_bounds, "new-layer", target)
    canvas_result = edit_image.get_layers()[0]
    assert (canvas_result.get_width(), canvas_result.get_height()) == (40, 30)
    assert not any(pixels(canvas_result)[:40 * 5 * 4])
    assert any(pixels(canvas_result)[40 * 5 * 4:40 * 6 * 4])
    edit_image.remove_layer(canvas_result)
    native.apply_edited_image(PNG, edit_image, "canvas edit", settings.MODELS[0],
                              canvas_bounds, "replace-layer", target)
    assert (target.get_width(), target.get_height()) == (20, 10)
    assert target.get_offsets() == (True, 7, 9) and len(edit_image.get_layers()) == 2
    print("PASS: selected-layer/canvas export, new result, replacement preserving properties", base_type)

opaque_image = Gimp.Image.new(20, 20, Gimp.ImageBaseType.RGB)
images.append(opaque_image)
opaque_layer = Gimp.Layer.new(opaque_image, "No alpha", 20, 20, Gimp.ImageType.RGB_IMAGE,
                             100, Gimp.LayerMode.NORMAL)
opaque_layer.fill(Gimp.FillType.WHITE)
opaque_image.insert_layer(opaque_layer, None, 0)
assert not opaque_layer.has_alpha()
native.apply_edited_image(PNG, opaque_image, "padding fixture", settings.MODELS[0],
                          (0, 0, 20, 20), "replace-layer", opaque_layer)
assert opaque_layer.has_alpha()
assert not any(pixels(opaque_layer)[:20 * 5 * 4])
assert any(pixels(opaque_layer)[20 * 5 * 4:20 * 6 * 4])
print("PASS: transparent padding, automatic alpha, and selection-aware replacement")

for input_mode, output_mode in (("invalid", "new-layer"), ("generate", "replace-layer"),
                                ("layer", "new-layer"), ("canvas", "new-layer")):
    try:
        native.editing_options(None, input_mode, output_mode)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Invalid editing options must fail explicitly.")
print("PASS: editing input and output validation")

with tempfile.TemporaryDirectory(prefix="gimpml-smoke-") as directory:
    fake = Path(directory) / "fake-python"
    fake.write_text(
        f"#!{ROOT}/.venv/bin/python\nimport json,sys\nfrom pathlib import Path\n"
        "payload=json.load(sys.stdin)\n"
        "assert 'input' not in payload\n"
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

    fake.write_text(
        f"#!{ROOT}/.venv/bin/python\nimport json,sys\nfrom pathlib import Path\n"
        "payload=json.load(sys.stdin)\n"
        "source=Path(payload['input'])\nassert source.is_absolute()\n"
        "assert source.read_bytes().startswith(b'\\x89PNG')\n"
        "import struct\n"
        "expected=(8,4) if payload['request']['prompt']=='offline edit' else (100,80)\n"
        "assert struct.unpack('>II',source.read_bytes()[16:24])==expected\n"
        f"Path(payload['output']).write_bytes({PNG!r})\n"
        "print(json.dumps({'status':'success','width':8,'height':4}))\n")
    assert native.Worker(ROOT, fake, 10).run({"prompt": "offline edit"}, PNG) == PNG
    native.runtime = lambda: (ROOT, fake)
    registered = registered_procedures["layer"]
    config = registered.create_config()
    config.set_property("prompt", "offline edit")
    config.set_property("output-mode", "new-layer")
    config.set_property("allow-image-upload", False)
    target = images[0].get_layers()[0]
    images[0].set_selected_layers([target])
    before_count = len(images[0].get_layers())
    result = native.run(registered, Gimp.RunMode.NONINTERACTIVE, images[0], [target], config, None)
    assert result.index(0) == Gimp.PDBStatusType.EXECUTION_ERROR
    assert len(images[0].get_layers()) == before_count
    config.set_property("allow-image-upload", True)
    result = native.run(registered, Gimp.RunMode.NONINTERACTIVE, images[0], [target], config, None)
    assert result.index(0) == Gimp.PDBStatusType.SUCCESS
    assert len(images[0].get_layers()) == before_count + 1
    registered = registered_procedures["canvas"]
    config = registered.create_config()
    config.set_property("prompt", "offline canvas edit")
    config.set_property("allow-image-upload", True)
    config.set_property("output-mode", "replace-layer")
    result = native.run(registered, Gimp.RunMode.NONINTERACTIVE, images[0], [target], config, None)
    assert result.index(0) == Gimp.PDBStatusType.SUCCESS
    assert len(images[0].get_layers()) == before_count + 1
    assert target.is_valid()
    native.runtime = actual_runtime
    print("PASS: offline editing callback, upload consent, and worker input transport")
    try:
        native.Worker(ROOT, ROOT / ".venv/bin/python", 10).run({"prompt": ""})
    except RuntimeError as error:
        assert str(error) == "Enter an image-generation prompt."
    else:
        raise AssertionError("The actual backend must reject an empty prompt without API calls.")
    print("PASS: actual backend interpreter and sanitized validation error")

    GimpUi.init("gimpml-smoke")
    actual_saved_status = native.credential_request(
        ROOT, ROOT / ".venv/bin/python", {"operation": "status"})
    assert set(actual_saved_status) == {"openrouter", "google"}
    assert all(type(value) is bool for value in actual_saved_status.values())
    print("PASS: real external Keychain worker returns saved status only")
    original_credentials = native.credential_request
    credential_calls = []

    def fake_credentials(root, python, payload):
        credential_calls.append(payload)
        return {"openrouter": True, "google": False}

    native.credential_request = fake_credentials
    settings_dialog = native.SettingsDialog(ROOT, fake)
    for provider, link in settings_dialog.key_help.items():
        assert link.get_uri() == settings.KEY_HELP_URLS[provider]
        assert link.get_visible()
        assert link.get_sensitive()

    def save_settings():
        if settings_dialog.busy:
            return GLib.SOURCE_CONTINUE
        assert settings_dialog.key_status["openrouter"].get_text() == "Key saved in Keychain."
        for entry in settings_dialog.entries.values():
            assert not entry.get_visibility()
            assert not entry.get_text()
        settings_dialog.entries["google"].set_text("offline-private-key")
        settings_dialog.remove["google"].set_active(True)
        settings_dialog.response(Gtk.ResponseType.OK)
        assert "not both" in settings_dialog.status.get_text()
        assert not settings_dialog.busy
        settings_dialog.remove["google"].set_active(False)
        settings_dialog.response(Gtk.ResponseType.OK)
        assert settings_dialog.busy
        assert not settings_dialog.entries["google"].get_text()
        return GLib.SOURCE_REMOVE

    GLib.timeout_add(50, save_settings)
    assert settings_dialog.run() == Gtk.ResponseType.OK
    assert credential_calls == [
        {"operation": "status"},
        {"operation": "save", "keys": {"google": "offline-private-key"}, "clear": []},
    ]
    settings_dialog.destroy()

    def failing_credentials(root, python, payload):
        raise RuntimeError("Unlock Keychain and allow access.")

    native.credential_request = failing_credentials
    settings_dialog = native.SettingsDialog(ROOT, fake)

    def dismiss_settings_error():
        if settings_dialog.busy:
            return GLib.SOURCE_CONTINUE
        assert settings_dialog.status.get_text() == "Unlock Keychain and allow access."
        assert settings_dialog.get_widget_for_response(Gtk.ResponseType.OK).get_sensitive()
        settings_dialog.response(Gtk.ResponseType.CANCEL)
        return GLib.SOURCE_REMOVE

    GLib.timeout_add(50, dismiss_settings_error)
    assert settings_dialog.run() == Gtk.ResponseType.CANCEL
    settings_dialog.destroy()
    native.credential_request = original_credentials
    print("PASS: masked settings, save transport, conflicting removal, and actionable errors")

    defaults = {**settings.preferences(), "prompt": "offline fixture", "fit_to_canvas": False}
    fake.write_text(
        f"#!{ROOT}/.venv/bin/python\nimport json,sys\nfrom pathlib import Path\n"
        "payload=json.load(sys.stdin)\n"
        f"Path(payload['output']).write_bytes({PNG!r})\n"
        "print(json.dumps({'status':'success','width':8,'height':4}))\n")
    dialog = native.GenerateDialog(ROOT, fake, settings, defaults)
    assert dialog.get_title() == "GIMP-ML - Generate Image"
    assert not dialog.edit_controls
    assert dialog.fit.get_parent() is not None
    assert dialog.upload_consent.get_parent() is None
    GLib.idle_add(lambda: dialog.response(Gtk.ResponseType.OK) or GLib.SOURCE_REMOVE)
    assert dialog.run() == Gtk.ResponseType.OK
    assert dialog.png == PNG
    dialog.destroy()
    print("PASS: native dialog asynchronous generation")

    for mode in ("layer", "canvas"):
        dialog = native.GenerateDialog(ROOT, fake, settings, defaults, images[0], [target], mode)
        title = "Edit Selected Layer" if mode == "layer" else "Edit Visible Canvas"
        assert dialog.get_title() == "GIMP-ML - " + title
        assert ("selected layer pixels" if mode == "layer" else "visible canvas") in dialog.disclosure.get_text()
        assert set(dialog.edit_controls) == {"output_mode"}
        assert dialog.fit.get_parent() is None
        assert dialog.upload_consent.get_parent() is not None
        assert not dialog.fit.get_sensitive()
        assert dialog.edit_controls["output_mode"].get_sensitive()
        assert not dialog.upload_consent.get_active()
        dialog.response(Gtk.ResponseType.OK)
        assert dialog.worker is None and "Allow image upload" in dialog.status.get_text()
        dialog.upload_consent.set_active(True)
        GLib.idle_add(lambda: dialog.response(Gtk.ResponseType.OK) or GLib.SOURCE_REMOVE)
        assert dialog.run() == Gtk.ResponseType.OK
        assert dialog.png == PNG and dialog.bounds is not None
        assert dialog.input_mode == mode and dialog.output_mode == "new-layer"
        dialog.destroy()
    print("PASS: editing dialog modes require explicit upload consent")

    fake.write_text(
        f"#!{ROOT}/.venv/bin/python\nimport json,sys\njson.load(sys.stdin)\n"
        "print(json.dumps({'status':'error','message':'Insufficient provider credits.'}))\n"
        "sys.exit(1)\n")
    def dismiss_error():
        if dialog.worker is not None or not dialog.status.get_text():
            return GLib.SOURCE_CONTINUE
        assert dialog.status.get_text() == "Insufficient provider credits."
        assert dialog.prompt.get_sensitive()
        if dialog.input_mode != "generate":
            assert dialog.edit_controls["output_mode"].get_sensitive()
            assert dialog.upload_consent.get_sensitive()
            assert not dialog.upload_consent.get_active()
        dialog.response(Gtk.ResponseType.CANCEL)
        return GLib.SOURCE_REMOVE

    for mode in ("generate", "canvas"):
        dialog = native.GenerateDialog(ROOT, fake, settings, defaults, images[0], [target], mode)
        dialog.upload_consent.set_active(mode != "generate")
        GLib.idle_add(lambda: dialog.response(Gtk.ResponseType.OK) or GLib.SOURCE_REMOVE)
        GLib.timeout_add(100, dismiss_error)
        assert dialog.run() == Gtk.ResponseType.CANCEL
        assert dialog.png is None and dialog.reference_png is None
        dialog.destroy()
    print("PASS: actionable provider errors leave inputs editable and insert no layer")

    fake.write_text(
        f"#!{ROOT}/.venv/bin/python\nimport json,sys,time\n"
        "json.load(sys.stdin)\ntime.sleep(30)\n")
    for mode in ("generate", "canvas"):
        dialog = native.GenerateDialog(ROOT, fake, settings, defaults, images[0], [target], mode)
        dialog.upload_consent.set_active(mode != "generate")
        GLib.idle_add(lambda: dialog.response(Gtk.ResponseType.OK) or GLib.SOURCE_REMOVE)
        GLib.timeout_add(200, lambda: dialog.response(Gtk.ResponseType.CANCEL) or GLib.SOURCE_REMOVE)
        started = time.monotonic()
        assert dialog.run() == Gtk.ResponseType.CANCEL
        assert dialog.png is None and dialog.worker is None and dialog.reference_png is None
        assert time.monotonic() - started < 5
        dialog.destroy()
    print("PASS: cancellation drains the worker and produces no image")

for image in images:
    image.clean_all()
    if image.get_id() != new_image.get_id():
        image.delete()

print("GIMP-ML OFFLINE SMOKE TEST PASSED")
