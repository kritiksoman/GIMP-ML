import matlab.engine
import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import cv2

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument(
    "--model", default="model/model_srresnet.pth", type=str, help="model path"
)
parser.add_argument(
    "--dataset", default="Set5", type=str, help="dataset name, Default: Set5"
)
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[
        shave_border : height - shave_border, shave_border : width - shave_border
    ]
    gt = gt[shave_border : height - shave_border, shave_border : width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


opt = parser.parse_args()
cuda = opt.cuda
eng = matlab.engine.start_matlab()

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model)["model"]

image_list = glob.glob("./testsets/" + opt.dataset + "/*.*")

avg_psnr_predicted = 0.0
avg_psnr_bicubic = 0.0
avg_elapsed_time = 0.0

for image_name in image_list:
    print("Processing ", image_name)
    im_gt_y = sio.loadmat(image_name)["im_gt_y"]
    im_b_y = sio.loadmat(image_name)["im_b_y"]
    im_l = sio.loadmat(image_name)["im_l"]

    im_gt_y = im_gt_y.astype(float)
    im_b_y = im_b_y.astype(float)
    im_l = im_l.astype(float)

    psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=opt.scale)
    avg_psnr_bicubic += psnr_bicubic

    im_input = im_l.astype(np.float32).transpose(2, 0, 1)
    im_input = im_input.reshape(
        1, im_input.shape[0], im_input.shape[1], im_input.shape[2]
    )
    im_input = Variable(torch.from_numpy(im_input / 255.0).float())

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    HR_4x = model(im_input)
    elapsed_time = time.time() - start_time
    avg_elapsed_time += elapsed_time

    HR_4x = HR_4x.cpu()

    im_h = HR_4x.data[0].numpy().astype(np.float32)

    im_h = im_h * 255.0
    im_h = np.clip(im_h, 0.0, 255.0)
    im_h = im_h.transpose(1, 2, 0).astype(np.float32)

    im_h_matlab = matlab.double((im_h / 255.0).tolist())
    im_h_ycbcr = eng.rgb2ycbcr(im_h_matlab)
    im_h_ycbcr = (
        np.array(im_h_ycbcr._data)
        .reshape(im_h_ycbcr.size, order="F")
        .astype(np.float32)
        * 255.0
    )
    im_h_y = im_h_ycbcr[:, :, 0]

    psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=opt.scale)
    avg_psnr_predicted += psnr_predicted

print("Scale=", opt.scale)
print("Dataset=", opt.dataset)
print("PSNR_predicted=", avg_psnr_predicted / len(image_list))
print("PSNR_bicubic=", avg_psnr_bicubic / len(image_list))
print("It takes average {}s for processing".format(avg_elapsed_time / len(image_list)))
