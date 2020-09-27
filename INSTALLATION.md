## GIMP-ML directory structure

```plaintext
|-- GIMP-ML
|   |-- gimp-plugins
|       |-- gimpenv
|       |-- weights
|       |-- .........
|       |-- <plugin files and folders>
|       |-- .........
|       |-- installGimpML.sh
|-- README.md
|-- INSTALLATION.md
```
## Installation Methods
Clone this repository: ```git clone https://github.com/kritiksoman/GIMP-ML.git``` <br>
### Manual Setup
[1] Install python 2.7 using ```sudo apt install python2-minimal``` or ```sudo apt install python-minimal``` if not already present. <br>
[2] Download and install pip ```wget https://bootstrap.pypa.io/get-pip.py ```, followed by ```python get-pip.py```, if not already present. <br>
[3] Install virtualenv package with ```python -m pip install --user virtualenv```. <br>
[4] Create a virtual environment ```gimpenv``` in the ```gimp-plugins``` folder using ```python -m virtualenv gimpenv```.<br>
[5] Activate the environment and install ```torchvision, "opencv-python<=4.3", numpy, future, torch, scipy, typing, enum, pretrainedmodels, requests``` using pip. <br>
[6] Open GIMP and go to Preferences -> Folders -> Plug-ins, add the folder gimp-plugins and close GIMP. <br>
[7] Download the weights folder from [link](https://drive.google.com/drive/folders/10IiBO4fuMiGQ-spBStnObbk9R-pGp6u8?usp=sharing) and move it inside ```gimp-plugins``` folder. <br>
[8] Allow the python scripts to be executable using ```chmod +x *.py``` in the ```gimp-plugins``` folder.<br>
[9] Run GIMP. <br>
Note: See [```installGimpML.sh```](https://github.com/kritiksoman/GIMP-ML/blob/master/INSTALLATION.md) if getting stuck.


### Using shell script and update tool
[1] Open terminal, go to GIMP-ML/gimp-plugins and run : ```bash installGimpML.sh```<br>
[2] Open GIMP and go to Preferences -> Folders -> Plug-ins, add the folder gimp-plugins and restart GIMP. <br>
[3] Go to Layer->GIMP-ML->update, click on ok with "update weights" set to yes and restart GIMP. (Weights ~ 1.5GB will be downloaded)<br>

