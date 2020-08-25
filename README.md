<img src="https://github.com/kritiksoman/tmp/blob/master/cover.png" width="1280" height="180"> <br>
# Semantics for GNU Image Manipulation Program
### :fast_forward: [YouTube](https://www.youtube.com/channel/UCzZn99R6Zh0ttGqvZieT4zw) :fast_forward: [Instagram](https://www.instagram.com/explore/tags/gimpml/) :fast_forward: [Manual](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual) :fast_forward: [Preprint](https://arxiv.org/abs/2004.13060) :fast_forward: [Medium](https://medium.com/@kritiksoman) :fast_forward: <br>
:star: :star: :star: :star: are welcome.<br>
Updates: <br>
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

# Update Steps
[1] Go to Layer->GIMP-ML->update, click on ok with "update weights" set to yes and restart GIMP. <br>

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
| facegen | CC BY-NC-SA 4.0 |
| deblur | BSD 3-clause |
| faceparse | MIT |
| deepcolor | MIT | 
| monodepth | MIT |
| super-resolution | MIT |
| deepmatting | Non-commercial purposes |
| deeplab | MIT |
| kmeans | BSD |
