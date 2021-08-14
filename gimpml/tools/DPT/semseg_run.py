import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose
from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from dpt_util.pallete import get_mask_pallete


def run(img, model_path, cpu_flag=False, model_type="dpt_hybrid", optimize=True):
    img = img / 255.0
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not cpu_flag else "cpu"
    )
    net_w = net_h = 480

    # load network
    if model_type == "dpt_large":
        model = DPTSegmentationModel(
            150,
            path=model_path,
            backbone="vitl16_384",
        )
    elif model_type == "dpt_hybrid":
        model = DPTSegmentationModel(
            150,
            path=model_path,
            backbone="vitb_rn50_384",
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        out = model.forward(sample)

        prediction = torch.nn.functional.interpolate(
            out, size=img.shape[:2], mode="bicubic", align_corners=False
        )
        prediction = torch.argmax(prediction, dim=1) + 1
        prediction = prediction.squeeze().cpu().numpy()

    mask = get_mask_pallete(prediction, "ade20k")
    mask = mask.convert("RGB")
    return np.array(mask)


# img = cv2.imread(r'D:\win\Users\Kritik Soman\Pictures\image.jpg')[:, :, ::-1]
# mask = run(img, r'C:\Users\Kritik Soman\GIMP-ML\weights\semseg\dpt_hybrid-ade20k-53898607.pt')
# cv2.imwrite("cache.png", mask)
