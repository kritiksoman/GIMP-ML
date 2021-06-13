This branch is under development. Dedicated for GIMP 3 and Python 3. <br>
<img src="https://github.com/kritiksoman/tmp/blob/master/cover.png" width="1280" height="180"> <br>

# Contribution 
[<img src="http://img.youtube.com/vi/vFFNp0xhEiU/0.jpg" width="800" height="600">](http://www.youtube.com/watch?v=vFFNp0xhEiU)<br> <br>
Welcome people interested in contribution !! 
Join us on Slack --> [<img src="https://woocommerce.com/wp-content/uploads/2015/02/Slack_RGB.png" width="130" height="50">](https://join.slack.com/t/gimp-mlworkspace/shared_invite/zt-rbaxvztx-GRvj941idw3sQ0trS686YA)<br>
Contribution guidelines will be available soon.<br>

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
