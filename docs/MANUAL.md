## In-painting (Edge-connect model)
Requires 2 inputs:<br>
[1] Image Layer. <br>
[2] Mask Layer containing mask of object to be removed. Background should be black (255,255,255) and object should be white (0,0,0). <br>
The mask layer should be created using paintbrush tool having 100 hardness and size as 15px. 
Both layers should be selected and then the plugin should be run from the GIMP-ML menu.

## Interpolate-frames
Requires 3 inputs:<br>
[1] Image Layer which will be the starting frame. <br>
[2] Image Layer which will be the ending frame. <br>
[3] Output Location: Folder where interpolated frames should be saved. <br>
Both layers should be selected and then the plugin should be run from the GIMP-ML menu.

## De-blur
Works on currently selected layer as input.


## De-haze
Works on currently selected layer as input.


## De-noise
Works on currently selected layer as input.


## Enlightening
Works on currently selected layer as input.


## MonoDepth
Works on currently selected layer as input.


## Semantic Segmentation
Works on currently selected layer as input containing any of the following: person, bird, cat, cow, dog, horse, sheep,  aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, and tv/monitor. <br> 


## Face Parsing
Works on currently selected layer as input containing only portrait image of a person.<br>


## Image Super-resolution
Requires the factor by which the image is to be enlarged as input.<br>
Set "Use as filter" to True if image size is medium/large in size (i.e., >~ 400pixels in height or width), otherwise you might run out of memory.<br>

## K-means Clustering
[1] Number of clusters/colors in output. <br>
[2] Use position: if (x,y) coordinates should be used as features for clustering. <br>

## Deep Image Matting
Requires 2 layers as input:
[1] Image Layer <br>
[2] Trimap Layer: Use RGB as [128,128,128] for boundaries, [255,255,255] for object and [0,0,0] for background. <br>
Example: <br>
![image1](https://github.com/kritiksoman/tmp/blob/master/trimap.png)<br>
Both layers should be selected and then the plugin should be run from the GIMP-ML menu.

## Deep Image Coloring
The image should be greyscale but the image mode should be RGB. This can be done from Image->Mode->RGB... <br>
Requires 2 layers as input:

[1] Image Layer <br>
[2] Color Mask Layer: A transparent RGB layer (with alpha channel) that contains (local points) dots of size 6 pixels specifying which color should be present at which location.<br>
Both layers should be selected and then the plugin should be run from the GIMP-ML menu.
