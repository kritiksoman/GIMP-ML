<img src="https://github.com/kritiksoman/tmp/blob/master/cover.png" width="1280" height="180"> <br>
# A.I. for GNU Image Manipulation Program
### [<img src="https://github.com/kritiksoman/tmp/blob/master/yt.png" width="70" height="50">](https://www.youtube.com/channel/UCzZn99R6Zh0ttGqvZieT4zw) [<img src="https://github.com/kritiksoman/tmp/blob/master/inst.png" width="50" height="50">](https://www.instagram.com/explore/tags/gimpml/) [<img src="https://github.com/kritiksoman/tmp/blob/master/arxiv.png" width="100" height="50">](https://arxiv.org/abs/2004.13060) [<img src="https://github.com/kritiksoman/tmp/blob/master/manual.png" width="100" height="50">](https://github.com/kritiksoman/GIMP-ML/blob/master/docs/MANUAL.md)[<img src="https://github.com/kritiksoman/tmp/blob/master/ref.png" width="100" height="50">](https://github.com/kritiksoman/GIMP-ML/blob/master/docs/REFERENCES.md) <br>
 
:star: :star: :star: :star: are welcome. This branch will no longer be updated and would only work with GIMP 2.10 and python 2.7 on mac and linux. Only GIMP3-ML branch will be updated in future which is targeted for python3 and GIMP3 (with windows and linux support) .<br>

[June 2] [GIMP3-ML](https://github.com/kritiksoman/GIMP-ML/tree/GIMP3-ML) branch development started.<br>
Old Updates: <br>
[January 9] Added image inpainting. (Existing users should be able to update.)<br>
[November 28] Added interpolate-frames.<br>
[October 31] Use super-resolution as a filter for medium/large images.<br>
[October 17] Added image enlightening.<br>
[September 27] Added Force CPU use button and minor bug fixes. <br>
[August 28] Added deep learning based dehazing and denoising. <br>
[August 25] Simplified installation and updating method. <br>
[August 2] Added deep matting and k-means. <br>
[July 17] MonoDepth and Colorization models have been updated. <br>

# Screenshot of Menu
![image1](https://github.com/kritiksoman/tmp/blob/master/screenshot.png)

# Installation Steps
## Install GIMP 2.10 with Python 2.7
### MacOS
[1] ```python``` command in terminal should point to GCC Apple LLVM Python 2.7. <br>
[2] Install [GIMP](https://download.gimp.org/pub/gimp/v2.10/) 2.10.22<br>
[3] Open GIMP,```Python-Fu``` should appear in the ```Filters``` menu.<br>

### Ubuntu
[1] Install GIMP 2.10 ```sudo apt install gimp```<br>
[2] Install Python 2.7 ```sudo apt install python2.7``` , followed by ```sudo apt install python python-cairo python-gobject-2```<br>
[3] Download Files: [gimp-python-debs.zip](https://drive.google.com/file/d/1g6Nea1-breldsV5TXSyFQIg_FYx2T5BL/view?usp=sharing) <br>
[4] Install the above downloaded files with ```sudo dpkg -i *.deb``` <br>
[5] Open GIMP,```Python-Fu``` should appear in the ```Filters``` menu.<br>

## Install GIMP-ML
[1] Clone this repository: git clone https://github.com/kritiksoman/GIMP-ML.git <br>
[2] Open terminal, go to GIMP-ML/gimp-plugins and run : <br>
    ```bash installGimpML.sh```<br>
[3] Open GIMP and go to Preferences -> Folders -> Plug-ins, add the folder gimp-plugins and restart GIMP. <br>
[4] Go to Layer->GIMP-ML->update, click on ok with "update weights" set to yes and restart GIMP. (Weights ~ 1.5GB will be downloaded)<br>

Manual install description if above is not working: [Link](https://github.com/kritiksoman/GIMP-ML/blob/master/INSTALLATION.md) <br>


# Citation
Please cite using the following bibtex entry:

```
@article{soman2020GIMPML,
  title={GIMP-ML: Python Plugins for using Computer Vision Models in GIMP},
  author={Soman, Kritik},
  journal={arXiv preprint arXiv:2004.13060},
  year={2020}
}
```

# Model Zoo
| Name | License | Dataset |
| ------------- |:-------------:| :-------------:| 
| [facegen](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#face-portrait-generation) | [CC BY-NC-SA 4.0](https://github.com/switchablenorms/CelebAMask-HQ#dataset-agreement) | CelebAMask-HQ |
| [deblur](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#de-blur) | [BSD 3-clause](https://github.com/VITA-Group/DeblurGANv2/blob/master/LICENSE) | GoPro |
| [faceparse](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#face-parsing) | [MIT](https://github.com/zllrunning/face-parsing.PyTorch/blob/master/LICENSE) | CelebAMask-HQ |
| [deepcolor](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#deep-image-coloring) | [MIT](https://github.com/junyanz/interactive-deep-colorization/blob/master/LICENSE) | ImageNet |
| [monodepth](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#monodepth) | [MIT](https://github.com/intel-isl/MiDaS/blob/master/LICENSE) | [Multiple](https://arxiv.org/pdf/1907.01341v3.pdf) |
| [super-resolution](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#image-super-resolution) | [MIT](https://github.com/twtygqyy/pytorch-SRResNet/blob/master/LICENSE) | ImageNet |
| [deepmatting](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#deep-image-matting) | [Non-commercial purposes](https://github.com/poppinace/indexnet_matting/blob/master/Adobe%20Deep%20Image%20Mattng%20Dataset%20License%20Agreement.pdf) | Adobe Deep Image Matting |
| [semantic-segmentation](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#semantic-segmentation) | MIT | COCO |
| [kmeans](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#k-means-clustering) | [BSD](https://github.com/scipy/scipy/blob/master/LICENSE.txt) | - |
| [deep-dehazing](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#de-haze) | [MIT](https://github.com/MayankSingal/PyTorch-Image-Dehazing/blob/master/LICENSE) | [Custom](https://sites.google.com/site/boyilics/website-builder/project-page) |
| [deep-denoising](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#de-noise) | [GPL3](https://github.com/SaoYan/DnCNN-PyTorch/blob/master/LICENSE) | BSD68 |
| [enlighten](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#enlightening) | [BSD](https://github.com/VITA-Group/EnlightenGAN/blob/master/License) | [Custom](https://arxiv.org/pdf/1906.06972.pdf) |
| [interpolate-frames](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#interpolate-frames) | [MIT](https://github.com/hzwer/arXiv2020-RIFE/blob/main/LICENSE) | [HD](https://arxiv.org/pdf/2011.06294.pdf) |
| [inpainting](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual#in-painting) | [CC BY-NC-SA 4.0](https://github.com/a-mos/High_Resolution_Image_Inpainting/blob/master/LICENSE.md) | [DIV2K](http://ceur-ws.org/Vol-2744/short18.pdf) |

# Contribution
If anyone is interested in contribution, then please see our GIMP3-ML branch--> [Link](https://github.com/kritiksoman/GIMP-ML/tree/GIMP3-ML)

# License
GIMP-ML is  [![MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/kritiksoman/GIMP-ML/blob/master/LICENSE.md), but each of the individual plugins follow the same license as the original model's.
