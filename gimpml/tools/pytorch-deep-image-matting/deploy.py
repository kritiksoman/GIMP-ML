import torch
import argparse
import torch.nn as nn
import deepmatting_net
import cv2
import os
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import time


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Super Res Example")
    parser.add_argument(
        "--size_h", type=int, default=320, help="height size of input image"
    )
    parser.add_argument(
        "--size_w", type=int, default=320, help="width size of input image"
    )
    parser.add_argument("--imgDir", type=str, required=True, help="directory of image")
    parser.add_argument(
        "--trimapDir", type=str, required=True, help="directory of trimap"
    )
    parser.add_argument("--cuda", action="store_true", help="use cuda?")
    parser.add_argument(
        "--resume", type=str, required=True, help="checkpoint that model resume from"
    )
    parser.add_argument(
        "--saveDir", type=str, required=True, help="where prediction result save to"
    )
    parser.add_argument("--alphaDir", type=str, default="", help="directory of gt")
    parser.add_argument(
        "--stage", type=int, required=True, choices=[0, 1, 2, 3], help="backbone stage"
    )
    parser.add_argument(
        "--not_strict", action="store_true", help="not copy ckpt strict?"
    )
    parser.add_argument(
        "--crop_or_resize",
        type=str,
        default="whole",
        choices=["resize", "crop", "whole"],
        help="how manipulate image before test",
    )
    parser.add_argument(
        "--max_size", type=int, default=1600, help="max size of test image"
    )
    args = parser.parse_args()
    print(args)
    return args


def gen_dataset(imgdir, trimapdir):
    sample_set = []
    img_ids = os.listdir(imgdir)
    img_ids.sort()
    cnt = len(img_ids)
    cur = 1
    for img_id in img_ids:
        img_name = os.path.join(imgdir, img_id)
        trimap_name = os.path.join(trimapdir, img_id)

        assert os.path.exists(img_name)
        assert os.path.exists(trimap_name)

        sample_set.append((img_name, trimap_name))

    return sample_set


def compute_gradient(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    grad = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    grad = cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    return grad


# inference once for image, return numpy
def inference_once(args, model, scale_img, scale_trimap, aligned=True):

    if aligned:
        assert scale_img.shape[0] == args.size_h
        assert scale_img.shape[1] == args.size_w

    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    scale_img_rgb = cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB)
    # first, 0-255 to 0-1
    # second, x-mean/std and HWC to CHW
    tensor_img = normalize(scale_img_rgb).unsqueeze(0)

    scale_grad = compute_gradient(scale_img)
    # tensor_img = torch.from_numpy(scale_img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    tensor_trimap = torch.from_numpy(
        scale_trimap.astype(np.float32)[np.newaxis, np.newaxis, :, :]
    )
    tensor_grad = torch.from_numpy(
        scale_grad.astype(np.float32)[np.newaxis, np.newaxis, :, :]
    )

    if args.cuda:
        tensor_img = tensor_img.cuda()
        tensor_trimap = tensor_trimap.cuda()
        tensor_grad = tensor_grad.cuda()
    # print('Img Shape:{} Trimap Shape:{}'.format(img.shape, trimap.shape))

    input_t = torch.cat((tensor_img, tensor_trimap / 255.0), 1)

    # forward
    if args.stage <= 1:
        # stage 1
        pred_mattes, _ = model(input_t)
    else:
        # stage 2, 3
        _, pred_mattes = model(input_t)
    pred_mattes = pred_mattes.data
    if args.cuda:
        pred_mattes = pred_mattes.cpu()
    pred_mattes = pred_mattes.numpy()[0, 0, :, :]
    return pred_mattes


# forward for a full image by crop method
def inference_img_by_crop(args, model, img, trimap):
    # crop the pictures, and forward one by one
    h, w, c = img.shape
    origin_pred_mattes = np.zeros((h, w), dtype=np.float32)
    marks = np.zeros((h, w), dtype=np.float32)

    for start_h in range(0, h, args.size_h):
        end_h = start_h + args.size_h
        for start_w in range(0, w, args.size_w):

            end_w = start_w + args.size_w
            crop_img = img[start_h:end_h, start_w:end_w, :]
            crop_trimap = trimap[start_h:end_h, start_w:end_w]

            crop_origin_h = crop_img.shape[0]
            crop_origin_w = crop_img.shape[1]

            # print("startH:{} startW:{} H:{} W:{}".format(start_h, start_w, crop_origin_h, crop_origin_w))

            if len(np.where(crop_trimap == 128)[0]) <= 0:
                continue

            # egde patch in the right or bottom
            if crop_origin_h != args.size_h or crop_origin_w != args.size_w:
                crop_img = cv2.resize(
                    crop_img, (args.size_w, args.size_h), interpolation=cv2.INTER_LINEAR
                )
                crop_trimap = cv2.resize(
                    crop_trimap,
                    (args.size_w, args.size_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            # inference for each crop image patch
            pred_mattes = inference_once(args, model, crop_img, crop_trimap)

            if crop_origin_h != args.size_h or crop_origin_w != args.size_w:
                pred_mattes = cv2.resize(
                    pred_mattes,
                    (crop_origin_w, crop_origin_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            origin_pred_mattes[start_h:end_h, start_w:end_w] += pred_mattes
            marks[start_h:end_h, start_w:end_w] += 1

    # smooth for overlap part
    marks[marks <= 0] = 1.0
    origin_pred_mattes /= marks
    return origin_pred_mattes


# forward for a full image by resize method
def inference_img_by_resize(args, model, img, trimap):
    h, w, c = img.shape
    # resize for network input, to Tensor
    scale_img = cv2.resize(
        img, (args.size_w, args.size_h), interpolation=cv2.INTER_LINEAR
    )
    scale_trimap = cv2.resize(
        trimap, (args.size_w, args.size_h), interpolation=cv2.INTER_LINEAR
    )

    pred_mattes = inference_once(args, model, scale_img, scale_trimap)

    # resize to origin size
    origin_pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation=cv2.INTER_LINEAR)
    assert origin_pred_mattes.shape == trimap.shape
    return origin_pred_mattes


# forward a whole image
def inference_img_whole(args, model, img, trimap):
    h, w, c = img.shape
    new_h = min(args.max_size, h - (h % 32))
    new_w = min(args.max_size, w - (w % 32))

    # resize for network input, to Tensor
    scale_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale_trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pred_mattes = inference_once(args, model, scale_img, scale_trimap, aligned=False)

    # resize to origin size
    origin_pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation=cv2.INTER_LINEAR)
    assert origin_pred_mattes.shape == trimap.shape
    return origin_pred_mattes


def main():

    print("===> Loading args")
    args = get_args()

    print("===> Environment init")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    model = net.VGG16(args)
    ckpt = torch.load(args.resume)
    if args.not_strict:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt["state_dict"], strict=True)

    if args.cuda:
        model = model.cuda()

    print("===> Load dataset")
    dataset = gen_dataset(args.imgDir, args.trimapDir)

    mse_diffs = 0.0
    sad_diffs = 0.0
    cnt = len(dataset)
    cur = 0
    t0 = time.time()
    for img_path, trimap_path in dataset:
        img = cv2.imread(img_path)
        trimap = cv2.imread(trimap_path)[:, :, 0]

        assert img.shape[:2] == trimap.shape[:2]

        img_info = (img_path.split("/")[-1], img.shape[0], img.shape[1])

        cur += 1
        print("[{}/{}] {}".format(cur, cnt, img_info[0]))

        with torch.no_grad():
            torch.cuda.empty_cache()

            if args.crop_or_resize == "whole":
                origin_pred_mattes = inference_img_whole(args, model, img, trimap)
            elif args.crop_or_resize == "crop":
                origin_pred_mattes = inference_img_by_crop(args, model, img, trimap)
            else:
                origin_pred_mattes = inference_img_by_resize(args, model, img, trimap)

        # only attention unknown region
        origin_pred_mattes[trimap == 255] = 1.0
        origin_pred_mattes[trimap == 0] = 0.0

        # origin trimap
        pixel = float((trimap == 128).sum())

        # eval if gt alpha is given
        if args.alphaDir != "":
            alpha_name = os.path.join(args.alphaDir, img_info[0])
            assert os.path.exists(alpha_name)
            alpha = cv2.imread(alpha_name)[:, :, 0] / 255.0
            assert alpha.shape == origin_pred_mattes.shape

            # x1 = (alpha[trimap == 255] == 1.0).sum() # x3
            # x2 = (alpha[trimap == 0] == 0.0).sum() # x5
            # x3 = (trimap == 255).sum()
            # x4 = (trimap == 128).sum()
            # x5 = (trimap == 0).sum()
            # x6 = trimap.size # sum(x3,x4,x5)
            # x7 = (alpha[trimap == 255] < 1.0).sum() # 0
            # x8 = (alpha[trimap == 0] > 0).sum() #

            # print(x1, x2, x3, x4, x5, x6, x7, x8)
            # assert(x1 == x3)
            # assert(x2 == x5)
            # assert(x6 == x3 + x4 + x5)
            # assert(x7 == 0)
            # assert(x8 == 0)

            mse_diff = ((origin_pred_mattes - alpha) ** 2).sum() / pixel
            sad_diff = np.abs(origin_pred_mattes - alpha).sum()
            mse_diffs += mse_diff
            sad_diffs += sad_diff
            print("sad:{} mse:{}".format(sad_diff, mse_diff))

        origin_pred_mattes = (origin_pred_mattes * 255).astype(np.uint8)
        res = origin_pred_mattes.copy()

        # only attention unknown region
        res[trimap == 255] = 255
        res[trimap == 0] = 0

        if not os.path.exists(args.saveDir):
            os.makedirs(args.saveDir)
        cv2.imwrite(os.path.join(args.saveDir, img_info[0]), res)

    print("Avg-Cost: {} s/image".format((time.time() - t0) / cnt))
    if args.alphaDir != "":
        print("Eval-MSE: {}".format(mse_diffs / cur))
        print("Eval-SAD: {}".format(sad_diffs / cur))


if __name__ == "__main__":
    main()
