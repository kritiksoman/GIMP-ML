We are currently refactoring the code, so expect small changes to be made to this.  

## GIMP3-ML directory structure
```plaintext
|-- gimpenv3 (Environment created with site-packages containing GIMP-ML folder when install.bat is run)
|-- GIMP-ML (Used for development.)
|   |-- gimpml
|   |-- __init__.py
|       |-- plugins
|           |-- .........
|           |-- monodepth
|               |-- monodepth.py
|           |-- enlighten
|               |-- enlighten.py
|           |-- .........
|       |-- tools
|         |-- .........
|         |-- MiDaS
|         |-- monodepth.py
|         |-- EnlightenGAN
|         |-- enlighten.py
|         |-- gimp_ml_config.pkl
|         |-- .........
|   |-- README.md
|   |-- CONTRIBUTION.md
|   |-- MANIFEST.in
|   |-- setup.py
|   |-- install.bat
```

## Weights location and cache file
This folder is present in user home directory. <br>
```plaintext
|-- GIMP-ML
|   |-- weights
|       |-- ........
|       |-- colorize
|           |-- caffemodel.pth
|       |-- deepdehaze
|           |-- dehazer.pth
|       |-- ........
|   |-- gimp_ml_run.pkl
```


## Plugin workflow
1> GIMP points to gimpenv3\Lib\site-packages\gimpml\plugins folder. <br>
2> Which we run a plugin, GIMP will call for example gimpenv3\Lib\site-packages\gimpml\plugins\monodepth\monodepth.py. <br>
3> This file contains UI part and will save selected layers to home folder/ GIMP-ML, and start a subprocess which will run gimpenv3\Lib\site-packages\gimpml\tools\monodepth.py and save result in same folder. <br>
4> When the subprocess is completed, the results are loaded back into GIMP. <br>

## Versions
GIMP-2.99.6, Python 3.8, Pytorch 1.8

## Creating new plugins
1> Open Pycharm/Spyder and point to ```|-- GIMP-ML (Used for development.)``` folder. <br>
2> Put model code in a folder and corresponding inference function into a separate python file such as tools/MiDaS folder and tools/monodepth.py. (See these examples.) <br>
3> Put these two in GIMP-ML/tools folder. <br>
4> Create corresponding plugin file in GIMP-ML/plugins folder. Example: GIMP-ML/plugins/monodepth/monodepth.py. <br>
5> Import corresponding inference function in gimpml/__init__.py so that the same function can be used in python package. Example: 
```Python
import cv2
import gimpml
face = cv2.imread('sampleinput/face.png')
out = gimpml.depth(face[:, :, ::-1])
cv2.imwrite('output/tmp-depth.png', out[:, :, ::-1])
```
6> With current location as GIMP-ML (contains setup.py), run ```pip install .```. The newly added plugins should move into site-packages folder. <br>
7> Restart GIMP to see the plugin in GIMP menu. <br>

## Issues
Join us on slack  [<img src="https://woocommerce.com/wp-content/uploads/2015/02/Slack_RGB.png" width="130" height="50">](https://join.slack.com/t/gimp-mlworkspace/shared_invite/zt-rbaxvztx-GRvj941idw3sQ0trS686YA)<br>  and ask doubts in ```dev-setup-issues``` channel. 
