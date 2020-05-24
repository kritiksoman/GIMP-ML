from _util import *

add_gimpenv_to_pythonpath()

from gimpfu import register, main, gimp
import gimpfu as gfu
import PIL.Image as pil
import torch
import torch.hub
from torchvision import transforms
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm


@tqdm_as_gimp_progress("Downloading model")
def load_model(device):
    repo = "valgur/monodepth2"
    pretrained_model = "mono+stereo_640x192"
    encoder = torch.hub.load(repo, "ResnetEncoder", pretrained_model, map_location=device)
    depth_decoder = torch.hub.load(repo, "DepthDecoder", pretrained_model, map_location=device)
    encoder.to(device)
    depth_decoder.to(device)
    return depth_decoder, encoder


@handle_alpha
@torch.no_grad()
def monodepth(input_image, device="cuda"):
    h, w, d = input_image.shape
    assert d == 3, "Input image must be RGB"

    # LOADING PRETRAINED MODEL
    depth_decoder, encoder = load_model(device)

    input_image = pil.fromarray(input_image)
    original_width, original_height = input_image.size
    input_image = input_image.resize((encoder.feed_width, encoder.feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    input_image = input_image.to(device)

    # PREDICTION
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Convert to colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im


def process(gimp_img, layer):
    gimp.progress_init("(Using {}) Generating disparity map for {}...".format(
        "GPU" if default_device().type == "cuda" else "CPU",
        layer.name
    ))
    img = layer_to_numpy(layer)
    result = monodepth(img, default_device())
    numpy_to_layer(result, gimp_img, layer.name + ' monodepth')


register(
    "MonoDepth",
    "MonoDepth",
    "Generate monocular disparity map based on deep learning.",
    "Kritik Soman",
    "",
    "2020",
    "MonoDepth...",
    "RGB*",
    [
        (gfu.PF_IMAGE, "image", "Input image", None),
        (gfu.PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    process,
    menu="<Image>/Layer/GIML-ML")

main()
