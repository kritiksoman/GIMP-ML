from _util import *

add_gimpenv_to_pythonpath()

from gimpfu import register, main, gimp
import gimpfu as gfu
import numpy as np
import torch
import torch.hub
from PIL import Image
from torchvision import transforms

colors = np.array([
    [0, 0, 0],
    [204, 0, 0],
    [0, 255, 255],
    [51, 255, 255],
    [51, 51, 255],
    [204, 0, 204],
    [204, 204, 0],
    [102, 51, 0],
    [255, 0, 0],
    [0, 204, 204],
    [76, 153, 0],
    [102, 204, 0],
    [255, 255, 0],
    [0, 0, 153],
    [255, 153, 51],
    [0, 51, 0],
    [0, 204, 0],
    [0, 0, 204],
    [255, 51, 153]
], dtype=np.uint8)


@tqdm_as_gimp_progress("Downloading model")
def load_model(device):
    net = torch.hub.load('valgur/face-parsing.PyTorch', 'BiSeNet', pretrained=True, map_location=device)
    net.to(device)
    return net


@handle_alpha
@torch.no_grad()
def faceparse(input_image, device="cuda"):
    h, w, d = input_image.shape
    assert d == 3, "Input image must be RGB"
    net = load_model(device)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    input_image = Image.fromarray(input_image)
    img = input_image.resize((512, 512), Image.BICUBIC)
    img = to_tensor(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)

    out = net(img)[0]

    result = out.squeeze(0).argmax(0).byte().cpu().numpy()
    result = Image.fromarray(result).resize(input_image.size)
    result.putpalette(colors.tobytes())
    result = np.array(result.convert('RGB'))
    return result


def process(gimp_img, layer):
    gimp.progress_init("(Using {}) Running face parse for {}...".format(
        "GPU" if default_device().type == "cuda" else "CPU",
        layer.name
    ))
    img = layer_to_numpy(layer)
    result = faceparse(img, default_device())
    numpy_to_layer(result, gimp_img, layer.name + ' faceparse')


register(
    "faceparse",
    "faceparse",
    "Running face parse.",
    "Kritik Soman",
    "",
    "2020",
    "faceparse...",
    "RGB*",
    [
        (gfu.PF_IMAGE, "image", "Input image", None),
        (gfu.PF_DRAWABLE, "drawable", "Input drawable", None)
    ],
    [],
    process,
    menu="<Image>/Layer/GIML-ML")

main()
