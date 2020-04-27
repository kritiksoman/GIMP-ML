# GIMP-ML
[![MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/kritiksoman/GIMP-ML/blob/master/LICENSE) <br>

Set of Machine Learning Python plugins for GIMP. 

The plugins have been tested with GIMP 2.10 on the following machines: <br>
[1] macOS Catalina 10.15.3 <br>
[2] ubuntu 18.04 LTS

# Screenshot of Menu
![image1](https://github.com/kritiksoman/GIMP-ML/blob/master/screenshot.png)


# Installation Steps
[1] Install [GIMP](https://www.gimp.org/downloads/).<br>
[2] Clone this repository: git clone https://github.com/kritiksoman/GIMP-ML.git <br>
[3] Open GIMP and go to Preferences -> Folders -> Plug-ins, add the folder gimp-plugins and close GIMP. <br>
[4] Download the [weights](https://drive.google.com/open?id=1mqzDnxtXQ75lVqlQ8tUeua68lDqUgUVe) and save it in gimp-plugins folder. <br>
[5] Open terminal and run : <br>
    ```bash moveWeights.sh ```
    <br>
    ```bash installGimpML-mac.sh```<br>
[6] Open GIMP.

# Demo videos on YouTube
[![](http://img.youtube.com/vi/U1CieWi--gc/0.jpg)](http://www.youtube.com/watch?v=U1CieWi--gc "") <br>
[![](http://img.youtube.com/vi/HeBgWcXFQpI/0.jpg)](http://www.youtube.com/watch?v=HeBgWcXFQpI "") <br>
[![](http://img.youtube.com/vi/adgHtu4chyU/0.jpg)](http://www.youtube.com/watch?v=adgHtu4chyU "") <br>
[![](http://img.youtube.com/vi/q9Ny5XqIUKk/0.jpg)](http://www.youtube.com/watch?v=q9Ny5XqIUKk "") <br>
[![](http://img.youtube.com/vi/thS8VqPvuhE/0.jpg)](http://www.youtube.com/watch?v=thS8VqPvuhE "") <br>
[![](http://img.youtube.com/vi/kXYsWvOB4uk/0.jpg)](http://www.youtube.com/watch?v=kXYsWvOB4uk "") <br>
[![](http://img.youtube.com/vi/HVwISLRow_0/0.jpg)](http://www.youtube.com/watch?v=HVwISLRow_0 "")

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
[5] https://github.com/richzhang/colorization <br>
[6] https://github.com/twtygqyy/pytorch-SRResNet
