from _util import *

add_gimpenv_to_pythonpath()

from gimpfu import register, main, gimp
import gimpfu as gfu
import numpy as np
import torch
import torch.hub
from PIL import Image
from torchvision import transforms


@tqdm_as_gimp_progress("Downloading model")
def load_model(device):
    model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def segment(input_image, device="cuda"):
    h, w, d = input_image.shape
    assert d == 3, "Input image must be RGB"
    
    model = load_model(device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    h, w, _ = input_image.shape
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)

    output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)

    # apply a color palette, selecting a color for each class
    palette = np.array([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = np.arange(21)[:, None] * palette
    colors = (colors % 255).astype(np.uint8)
    result = Image.fromarray(output_predictions.byte().cpu().numpy()).resize((w, h))
    result.putpalette(colors.tobytes())
    return np.array(result.convert('RGB'))


def process(img, layer):
    gimp.progress_init("(Using {}) Generating semantic segmentation map for {}...".format(
        "GPU" if default_device().type == "cuda" else "CPU",
        layer.name
    ))
    rgb, alpha = split_alpha(layer_to_numpy(layer))
    result = segment(rgb, default_device())
    result = merge_alpha(result, alpha)
    numpy_to_layer(result, img, layer.name + ' segmented')


register(
    "deeplabv3",
    "deeplabv3",
    "Generate semantic segmentation map based on deep learning.",
    "Kritik Soman",
    "GIMP-ML",
    "2020",
    "deeplabv3...",
    "RGB*",
    [
        (gfu.PF_IMAGE, "image", "Input image", None),
        (gfu.PF_DRAWABLE, "drawable", "Input drawable", None)
    ],
    [],
    process,
    menu="<Image>/Layer/GIML-ML")

main()
