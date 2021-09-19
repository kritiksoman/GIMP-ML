import gi
gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gimp, GimpUi, GObject, GLib, Gio, Gtk
import gettext

_ = gettext.gettext


def show_dialog(message, title, icon="logo", image_paths=None):
    use_header_bar = Gtk.Settings.get_default().get_property("gtk-dialogs-use-header")
    dialog = GimpUi.Dialog(use_header_bar=use_header_bar, title=_(title))
    # Add buttons
    dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
    dialog.add_button("_OK", Gtk.ResponseType.APPLY)
    vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, homogeneous=False, spacing=10)
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
    logo = Gtk.Image.new_from_file(image_paths[icon])
    # vbox.pack_start(logo, False, False, 1)
    grid.attach(logo, 0, 0, 1, 1)
    logo.show()
    # Show message
    label = Gtk.Label(label=_(message))
    # vbox.pack_start(label, False, False, 1)
    grid.attach(label, 1, 0, 1, 1)
    label.show()
    dialog.show()
    dialog.run()
    return


def save_image(image, drawable, file_path):
    interlace, compression = 0, 2
    Gimp.get_pdb().run_procedure(
        "file-png-save",
        [
            GObject.Value(Gimp.RunMode, Gimp.RunMode.NONINTERACTIVE),
            GObject.Value(Gimp.Image, image),
            GObject.Value(GObject.TYPE_INT, 1),
            GObject.Value(
                Gimp.ObjectArray, Gimp.ObjectArray.new(Gimp.Drawable, drawable, 0)
            ),
            GObject.Value(
                Gio.File,
                Gio.File.new_for_path(file_path),
            ),
            GObject.Value(GObject.TYPE_BOOLEAN, interlace),
            GObject.Value(GObject.TYPE_INT, compression),

            GObject.Value(GObject.TYPE_BOOLEAN, True),
            GObject.Value(GObject.TYPE_BOOLEAN, True),
            GObject.Value(GObject.TYPE_BOOLEAN, False),
            GObject.Value(GObject.TYPE_BOOLEAN, True),
        ],
    )


def N_(message):
    return message