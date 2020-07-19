# GIMP-ML
## [Wiki](https://github.com/kritiksoman/GIMP-ML/wiki) [Instagram](https://www.instagram.com/kritiksoman/) [Medium](https://medium.com/@kritiksoman) [Preprint](https://arxiv.org/abs/2004.13060) <br>
Semantics for GNU Image Manipulation Program. <br>
Updates: <br>
[July 17] MonoDepth and Colorization models have been updated. <br>

# Screenshot of Menu
![image1](https://github.com/kritiksoman/GIMP-ML/blob/master/screenshot.png)

# Installation Steps
[1] Install [GIMP](https://www.gimp.org/downloads/).<br>
[2] Clone this repository: git clone https://github.com/kritiksoman/GIMP-ML.git <br>
[3] Open GIMP and go to Preferences -> Folders -> Plug-ins, add the folder gimp-plugins and close GIMP. <br>
[4] Download [weights.zip](https://drive.google.com/open?id=1mqzDnxtXQ75lVqlQ8tUeua68lDqUgUVe) (1.22 GB) and save it in gimp-plugins folder. <br>
[5] Open terminal and run : <br>
    ```bash installGimpML.sh```
    <br>
    ```bash moveWeights.sh ```<br>
[6] Open GIMP.


# Common Issues
[1] No output on running plugin: Please right click on layer and remove alpha channel before using plugins. <br>
[2] GIMP-ML menu not visible: Do following and restart GIMP.<br>
```
sudo apt install gimp-python
cd gimp-plugins
chmod -x *
chmod +x *.py
```
[3] colorize plugin not working: Switch to grayscale mode before running plugin. (Image->Mode->Grayscale)

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
