## In-painting
Requires 2 inputs:<br>
![image1](https://github.com/kritiksoman/tmp/blob/master/inpainting.png)<br>
[1] Image Layer. <br>
[2] Mask Layer containing mask of object to be removed. Background should be black (255,255,255) and object should be white (0,0,0). <br>

## Interpolate-frames
Requires 3 inputs:<br>
![image1](https://github.com/kritiksoman/tmp/blob/master/interpolate-frames.png)<br>
[1] Image Layer which will be the starting frame. <br>
[2] Image Layer which will be the ending frame. <br>
[3] Output Location: Folder where interpolated frames should be saved. <br>

## De-blur
Works on currently selected layer as input.
![image1](https://github.com/kritiksoman/tmp/blob/master/deblur.png)<br>

## De-haze
Works on currently selected layer as input.
![image1](https://github.com/kritiksoman/tmp/blob/master/dehaze.png)<br>

## De-noise
Works on currently selected layer as input.
![image1](https://github.com/kritiksoman/tmp/blob/master/denoise.png)<br>

## Enlightening
Works on currently selected layer as input.
![image1](https://github.com/kritiksoman/tmp/blob/master/enlighten.png)<br>

## MonoDepth
Works on currently selected layer as input.
![image1](https://github.com/kritiksoman/tmp/blob/master/monodepth.png)<br>

## Semantic Segmentation
Works on currently selected layer as input containing any of the following: person, bird, cat, cow, dog, horse, sheep,  aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, and tv/monitor. <br> 
![image1](https://github.com/kritiksoman/tmp/blob/master/semseg.png)<br>

## Face Parsing
Works on currently selected layer as input containing only portrait image of a person.<br>
![image1](https://github.com/kritiksoman/tmp/blob/master/faceparse.png)<br>

## Face Portrait Generation
Requires 3 layers as input: 
![image1](https://github.com/kritiksoman/tmp/blob/master/facegen.png)<br>
[1] Image Layer containing only the portrait. <br>
[2] Original Mask Layer obtained by using faceparse on the image layer. <br>
[3] Modified Mask Layer obtained by duplicating the original mask layer and modifying it using paintbrush tool. <br>

## Image Super-resolution
Requires the factor by which the image is to be enlarged as input.<br>
![image1](https://github.com/kritiksoman/tmp/blob/master/super-resolution.png)<br>
Set "Use as filter" to True if image size is medium/large in size (i.e., >~ 400pixels in height or width), otherwise you might run out of memory.<br>

## K-means Clustering
Requires 3 inputs:<br>
![image1](https://github.com/kritiksoman/tmp/blob/master/kmeans.png)<br>
[1] Image Layer. <br>
[2] Number of clusters/colors in output. <br>
[3] Use position: if (x,y) coordinates should be used as features for clustering. <br>

## Deep Image Matting
Requires 2 layers as input: 
![image1](https://github.com/kritiksoman/tmp/blob/master/deepmatting.png)<br>
[1] Image Layer <br>
[2] Trimap Layer: Use RGB as [128,128,128] for boundaries, [255,255,255] for object and [0,0,0] for background. <br>
Example: <br>
![image1](https://github.com/kritiksoman/tmp/blob/master/trimap.png)<br>

## Deep Image Coloring
The image should be greyscale but the image mode should be RGB. This can be done from Image->Mode->RGB... <br>
Requires 2 layers as input:
![image1](https://github.com/kritiksoman/tmp/blob/master/deepcolor.png)<br>
[1] Image Layer <br>
[2] Color Mask Layer: A transparent RGB layer (with alpha channel) that contains (local points) dots of size 6 pixels specifying which color should be present at which location.<br>
Example: <br>
![image1](https://github.com/kritiksoman/tmp/blob/master/colormask.png)<br>
If the image and color mask layers are set to the same layer containing the image, the local points network will still give prediction. So the color mask layer is optional.