This branch is under development. Dedicated for GIMP 3 and Python 3. :star: :star: :star: :star: are welcome. <br>
<img src="https://github.com/kritiksoman/tmp/blob/master/cover.png" width="1280" height="180"> <br>

# Objectives
[1] Model Ensembling. <br>
[2] Deep learning inference package for different computer vision tasks. <br>
[3] Bridge gap between CV research work and real world data. <br>
[4] Add AI to routine image editing workflows. <br>

# Contribution 
[<img src="http://img.youtube.com/vi/vFFNp0xhEiU/0.jpg" width="800" height="600">](http://www.youtube.com/watch?v=vFFNp0xhEiU)<br> <br>
Welcome people interested in contribution !! 
Join us on Slack --> [<img src="https://woocommerce.com/wp-content/uploads/2015/02/Slack_RGB.png" width="130" height="50">](https://join.slack.com/t/gimp-mlworkspace/shared_invite/zt-rbaxvztx-GRvj941idw3sQ0trS686YA)<br>
Contribution guidelines available --> [Link](https://github.com/kritiksoman/GIMP-ML/blob/GIMP3-ML/CONTRIBUTION.md).<br>

# Screenshot of Menu
![image1](https://github.com/kritiksoman/GIMP-ML/blob/GIMP3-ML/screenshot.png)

# Installation Steps
[1] Install [GIMP](https://www.gimp.org/downloads/devel/) 2.99.6  (Only windows and linux) <br>
[2] Clone this repository: git clone https://github.com/kritiksoman/GIMP-ML.git <br>
[3] Change branch : <br>
```git checkout GIMP3-ML``` <br>
[3] On linux, run for GPU/CPU: <br>
```bash GIMP-ML/install.bat```<br>
On windows, run for CPU: <br>
```GIMP-ML\install.bat```<br>
On windows, run for GPU: <br>
```GIMP-ML\install.bat gpu```<br>
[4] Follow steps that are printed in terminal or cmd. <br>

### Windows <br>
[<img src="http://img.youtube.com/vi/Rc88_qHSEjc/0.jpg" width="400" height="300">](http://www.youtube.com/watch?v=Rc88_qHSEjc)<br> <br>

### Linux <br>
[<img src="http://img.youtube.com/vi/MUdUzxYDwaU/0.jpg" width="400" height="300">](http://www.youtube.com/watch?v=MUdUzxYDwaU)<br> <br>


# Use as a Python Package
```Python
import cv2
import gimpml

image = cv2.imread('sampleinput/img.png')
alpha = cv2.imread('sampleinput/alpha.png')

out = gimpml.kmeans(image)
cv2.imwrite('output/tmp-kmeans.jpg', out)

out = gimpml.deblur(image)
cv2.imwrite('output/tmp-deblur.jpg', out)

out = gimpml.deepcolor(image)
cv2.imwrite('output/tmp-deepcolor.jpg', out)

out = gimpml.dehaze(image)
cv2.imwrite('output/tmp-dehaze.jpg', out)

out = gimpml.denoise(image)
cv2.imwrite('output/tmp-denoise.jpg', out)

out = gimpml.matting(image, alpha)
cv2.imwrite('output/tmp-matting.png', out)  # save as png

out = gimpml.enlighten(image)
cv2.imwrite('output/tmp-enlighten.jpg', out)

```

# Model Zoo
| Name | License | Dataset |
| ------------- |:-------------:| :-------------:| 
| [deblur](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#de-blur) | [BSD 3-clause](https://github.com/VITA-Group/DeblurGANv2/blob/master/LICENSE) | GoPro |
| [faceparse](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#face-parsing) | [MIT](https://github.com/zllrunning/face-parsing.PyTorch/blob/master/LICENSE) | CelebAMask-HQ |
| [coloring](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#deep-image-coloring) | [MIT](https://github.com/junyanz/interactive-deep-colorization/blob/master/LICENSE) | ImageNet |
| [monodepth](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#monodepth) | [MIT](https://github.com/intel-isl/MiDaS/blob/master/LICENSE) | [Multiple](https://arxiv.org/pdf/1907.01341v3.pdf) |
| [super-resolution](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#image-super-resolution) | [MIT](https://github.com/twtygqyy/pytorch-SRResNet/blob/master/LICENSE) | ImageNet |
| [matting](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#deep-image-matting) | [Non-commercial purposes](https://github.com/poppinace/indexnet_matting/blob/master/Adobe%20Deep%20Image%20Mattng%20Dataset%20License%20Agreement.pdf) | Adobe Deep Image Matting |
| [semantic-segmentation](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#semantic-segmentation) | MIT | COCO |
| [kmeans](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#k-means-clustering) | [BSD](https://github.com/scipy/scipy/blob/master/LICENSE.txt) | - |
| [dehazing](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#de-haze) | [MIT](https://github.com/MayankSingal/PyTorch-Image-Dehazing/blob/master/LICENSE) | [Custom](https://sites.google.com/site/boyilics/website-builder/project-page) |
| [denoising](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#de-noise) | [GPL3](https://github.com/SaoYan/DnCNN-PyTorch/blob/master/LICENSE) | BSD68 |
| [enlighten](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#enlightening) | [BSD](https://github.com/VITA-Group/EnlightenGAN/blob/master/License) | [Custom](https://arxiv.org/pdf/1906.06972.pdf) |
| [interpolate-frames](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#interpolate-frames) | [MIT](https://github.com/hzwer/arXiv2020-RIFE/blob/main/LICENSE) | [HD](https://arxiv.org/pdf/2011.06294.pdf) |
| [inpainting](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#in-painting) | [CC BY-NC-SA 4.0](https://github.com/a-mos/High_Resolution_Image_Inpainting/blob/master/LICENSE.md) | [DIV2K](http://ceur-ws.org/Vol-2744/short18.pdf) |


