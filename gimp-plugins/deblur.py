from _util import *

add_gimpenv_to_pythonpath()

from gimpfu import register, main, gimp
import gimpfu as gfu
import torch.hub


@tqdm_as_gimp_progress("Downloading model")
def load_model(device):
    predictor = torch.hub.load('valgur/DeblurGANv2:python2', 'predictor', 'fpn_inception', device=device)
    return predictor


@handle_alpha
@torch.no_grad()
def deblur(img, device="cuda"):
    h, w, d = img.shape
    assert d == 3, "Input image must be RGB"
    predictor = load_model(device)
    pred = predictor(img, None)
    return pred


def process(gimp_img, layer):
    gimp.progress_init("(Using {}) Deblurring {}...".format(
        "GPU" if default_device().type == "cuda" else "CPU",
        layer.name
    ))
    img = layer_to_numpy(layer)
    result = deblur(img, default_device())
    numpy_to_layer(result, gimp_img, layer.name + ' deblurred')


register(
    "deblur",
    "deblur",
    "Running deblurring.",
    "Kritik Soman",
    "",
    "2020",
    "deblur...",
    "RGB*",
    [
        (gfu.PF_IMAGE, "image", "Input image", None),
        (gfu.PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    process,
    menu="<Image>/Layer/GIML-ML")

main()
