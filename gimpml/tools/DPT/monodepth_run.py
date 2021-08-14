import torch
import cv2
from torchvision.transforms import Compose
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import numpy as np


def run(
    img,
    model_path,
    cpu_flag=False,
    model_type="dpt_hybrid",
    optimize=True,
    kitti_crop=False,
    absolute_depth=False,
    bits=2,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not cpu_flag else "cpu"
    )
    img = img / 255.0

    # load network
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_kitti":
        net_w = 1216
        net_h = 352

        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid_nyu":
        net_w = 640
        net_h = 480

        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":  # Convolutional model
        net_w = net_h = 384

        model = MidasNet_large(model_path, non_negative=True)
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

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
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    if kitti_crop is True:
        height, width, _ = img.shape
        top = height - 352
        left = (width - 1216) // 2
        img = img[top : top + 352, left : left + 1216, :]

    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        if model_type == "dpt_hybrid_kitti":
            prediction *= 256

        if model_type == "dpt_hybrid_nyu":
            prediction *= 1000.0

    if absolute_depth:
        out = prediction
    else:
        depth_min = prediction.min()
        depth_max = prediction.max()

        max_val = (2 ** (8 * bits)) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (prediction - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(prediction.shape, dtype=prediction.dtype)

    return out


#
# img = cv2.imread(r'D:\win\Users\Kritik Soman\Pictures\image.jpg')[:, :, ::-1]
# bits = 2
#
# pred = run(img, r'C:\Users\Kritik Soman\GIMP-ML\weights\MiDaS\dpt_hybrid-midas-501f0c75.pt', bits=bits)
# # util.io.write_depth("cache", pred, bits=2, absolute_depth=False)
# if bits == 1:
#     cv2.imwrite("cache.png", pred.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
# elif bits == 2:
#     cv2.imwrite("cache.png", pred.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
