# GIMP-ML
Preprint: [Link](https://arxiv.org/abs/2004.13060) <br>
Set of Machine Learning Python plugins for GIMP. 

The plugins have been tested with GIMP 2.10 on the following machines: <br>
[1] macOS Catalina 10.15.4 <br>
[2] ubuntu 18.04 LTS <br>
[3] ubuntu 20.04 LTS <br>
[4] Debian GNU/Linux 10 (buster)

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

# Demo videos on YouTube
[<img src="http://img.youtube.com/vi/q9Ny5XqIUKk/0.jpg" width="400" height="300">](http://www.youtube.com/watch?v=q9Ny5XqIUKk)
[<img src="http://img.youtube.com/vi/kXYsWvOB4uk/0.jpg" width="400" height="300">](http://www.youtube.com/watch?v=kXYsWvOB4uk)

[<img src="http://img.youtube.com/vi/HVwISLRow_0/0.jpg" width="400" height="300">](http://www.youtube.com/watch?v=HVwISLRow_0)
[<img src="http://img.youtube.com/vi/U1CieWi--gc/0.jpg" width="400" height="300">](http://www.youtube.com/watch?v=U1CieWi--gc) 

[<img src="http://img.youtube.com/vi/HeBgWcXFQpI/0.jpg" width="400" height="300">](http://www.youtube.com/watch?v=HeBgWcXFQpI)
[<img src="http://img.youtube.com/vi/adgHtu4chyU/0.jpg" width="400" height="300">](http://www.youtube.com/watch?v=adgHtu4chyU) 

[<img src="http://img.youtube.com/vi/thS8VqPvuhE/0.jpg" width="400" height="300">](http://www.youtube.com/watch?v=thS8VqPvuhE) 

# Paper References
[1] Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (https://arxiv.org/abs/1609.04802) <br>
[2] DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better (https://arxiv.org/abs/1908.03826) <br>
[3] Digging into Self-Supervised Monocular Depth Prediction (https://arxiv.org/abs/1806.01260) <br>
[4] BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation (https://arxiv.org/abs/1808.00897) <br>
[5] MaskGAN: Towards Diverse and Interactive Facial Image Manipulation (https://arxiv.org/abs/1907.11922) <br>
[6] Perceptual Losses for Real-Time Style Transfer and Super-Resolution (https://cs.stanford.edu/people/jcjohns/eccv16/) <br>
[7] Rethinking Atrous Convolution for Semantic Image Segmentation (https://arxiv.org/abs/1706.05587) <br>

# Code References
The following have been ported : <br>
[1] https://github.com/switchablenorms/CelebAMask-HQ <br>
[2] https://github.com/TAMU-VITA/DeblurGANv2 <br>
[3] https://github.com/zllrunning/face-parsing.PyTorch <br>
[4] https://github.com/nianticlabs/monodepth2 <br>
[5] https://github.com/zeruniverse/neural-colorization <br>
[6] https://github.com/twtygqyy/pytorch-SRResNet

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
