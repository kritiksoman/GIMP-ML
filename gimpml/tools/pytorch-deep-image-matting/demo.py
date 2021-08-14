import torch
from argparse import Namespace
import net
import cv2
import os
import numpy as np
from deploy import inference_img_whole

# input file list
image_path = "boy-1518482_1920_12_img.png"
trimap_path = "boy-1518482_1920_12.png"
image = cv2.imread(image_path)
trimap = cv2.imread(trimap_path)
# print(trimap.shape)
trimap = trimap[:, :, 0]
# init model
args = Namespace(
    crop_or_resize="whole",
    cuda=True,
    max_size=1600,
    resume="model/stage1_sad_57.1.pth",
    stage=1,
)
model = net.VGG16(args)
ckpt = torch.load(args.resume)
model.load_state_dict(ckpt["state_dict"], strict=True)
model = model.cuda()

torch.cuda.empty_cache()
with torch.no_grad():
    pred_mattes = inference_img_whole(args, model, image, trimap)
pred_mattes = (pred_mattes * 255).astype(np.uint8)
pred_mattes[trimap == 255] = 255
pred_mattes[trimap == 0] = 0
# print(pred_mattes)
# cv2.imwrite('out.png', pred_mattes)

# import matplotlib.pyplot as plt
# plt.imshow(image)
# plt.show()
