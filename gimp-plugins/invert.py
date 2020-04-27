
from gimpfu import *

def invert(img, layer) :
    gimp.progress_init("Inverting " + layer.name + "...")
    pdb.gimp_undo_push_group_start(img)
    pdb.gimp_invert(layer)
    pdb.gimp_undo_push_group_end(img)

register(
    "Invert",
    "Invert",
    "Invert",
    "Kritik Soman",
    "Your Name",
    "2020",
    "Invert...",
    "*",      # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [   (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    invert, menu="<Image>/Filters/Enhance")

main()
