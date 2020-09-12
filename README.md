<img src="https://github.com/kritiksoman/tmp/blob/master/cover.png" width="1280" height="180"> <br>
# Semantics for GNU Image Manipulation Program
### :fast_forward: [YouTube](https://www.youtube.com/channel/UCzZn99R6Zh0ttGqvZieT4zw) :fast_forward: [Instagram](https://www.instagram.com/explore/tags/gimpml/) :fast_forward: [Manual](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual) :fast_forward: [Preprint](https://arxiv.org/abs/2004.13060) :fast_forward: [Medium](https://medium.com/@kritiksoman) :fast_forward: <br>
:star: :star: :star: :star: are welcome. New tools will be added and existing will be improved with time.<br>
Updates: <br>
[August 28] Added deep learning based dehazing and denoising (should be installed by updating). <br>
[August 25] Simplified installation and updating method. <br>
[August 2] Added deep matting and k-means. <br>
[July 17] MonoDepth and Colorization models have been updated. <br>

# Screenshot of Menu
![image1](https://github.com/kritiksoman/tmp/blob/master/screenshot.png)

# Installation Steps
[1] Install [GIMP](https://www.gimp.org/downloads/) 2.10.<br>
[2] Clone this repository: git clone https://github.com/kritiksoman/GIMP-ML.git <br>
[3] Open terminal, go to GIMP-ML/gimp-plugins and run : <br>
    ```bash installGimpML.sh```<br>
[4] Open GIMP and go to Preferences -> Folders -> Plug-ins, add the folder gimp-plugins and restart GIMP. <br>
[5] Go to Layer->GIMP-ML->update, click on ok with "update weights" set to yes and restart GIMP. (Weights ~ 1.5GB will be downloaded)<br>
Manual install description if above is not working: [Link](https://github.com/kritiksoman/GIMP-ML/blob/master/INSTALLATION.md) <br>

# Update Steps
[1] Go to Layer->GIMP-ML->update, click on ok with "update weights" set to NO and restart GIMP. <br>
[2] Go to Layer->GIMP-ML->update, click on ok with "update weights" set to YES and restart GIMP. <br>

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

# License
| Tools | License |
| ------------- |:-------------:| 
| facegen | [CC BY-NC-SA 4.0](https://github.com/switchablenorms/CelebAMask-HQ#dataset-agreement) |
| deblur | [BSD 3-clause](https://github.com/VITA-Group/DeblurGANv2/blob/master/LICENSE) |
| faceparse | [MIT](https://github.com/zllrunning/face-parsing.PyTorch/blob/master/LICENSE) |
| deepcolor | [MIT](https://github.com/junyanz/interactive-deep-colorization/blob/master/LICENSE) | 
| monodepth | [MIT](https://github.com/intel-isl/MiDaS/blob/master/LICENSE) |
| super-resolution | [MIT](https://github.com/twtygqyy/pytorch-SRResNet/blob/master/LICENSE) |
| deepmatting | [Non-commercial purposes](https://github.com/poppinace/indexnet_matting/blob/master/Adobe%20Deep%20Image%20Mattng%20Dataset%20License%20Agreement.pdf) |
| deeplab | MIT |
| kmeans | [BSD](https://github.com/scipy/scipy/blob/master/LICENSE.txt) |
| deep-dehazing | [MIT](https://github.com/MayankSingal/PyTorch-Image-Dehazing/blob/master/LICENSE) |
| deep-denoising | [GPL3](https://github.com/SaoYan/DnCNN-PyTorch/blob/master/LICENSE) |
