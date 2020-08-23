<img src="https://github.com/kritiksoman/tmp/blob/master/cover.png" width="1280" height="180"> <br>
# Semantics for GNU Image Manipulation Program
### [YouTube](https://www.youtube.com/channel/UCzZn99R6Zh0ttGqvZieT4zw) [Instagram](https://www.instagram.com/explore/tags/gimpml/) [Manual](https://github.com/kritiksoman/GIMP-ML/wiki/User-Manual) [Preprint](https://arxiv.org/abs/2004.13060) [Medium](https://medium.com/@kritiksoman)<br>

Updates: <br>
[Work in progress.] <br>
[August 23] Simplified installation and updating method. <br>
[August 2] Added deep matting and k-means. <br>
[July 17] MonoDepth and Colorization models have been updated. <br>

# Screenshot of Menu
![image1](https://github.com/kritiksoman/tmp/blob/master/screenshot.png)

# Installation Steps
[1] Install [GIMP](https://www.gimp.org/downloads/) 2.10.<br>
[2] Clone this repository: git clone https://github.com/kritiksoman/GIMP-ML.git <br>
[3] Open GIMP and go to Preferences -> Folders -> Plug-ins, add the folder gimp-plugins and close GIMP. <br>
[4] Open terminal and run : <br>
    ```bash installGimpML.sh```<br>
[5] Open GIMP and go to Layer->GIMP-ML->update and click on ok with update weights set to yes. <br>
[6] Restart GIMP.

# Update Steps
This will work if cloned after 24 August 2020 otherwise install again.<br>
[1] Open GIMP and go to Layer->GIMP-ML->update and click on ok with update weights set to yes. <br>
[2] Restart GIMP.


# Common Issues
[1] No output on running plugin: Please right click on layer and remove alpha channel before using plugins. <br>
[2] GIMP-ML menu not visible: Do following and restart GIMP.<br>
```
sudo apt install gimp-python
cd gimp-plugins
chmod -x *
chmod +x *.py
```

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
