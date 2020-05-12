import os 
baseLoc = os.path.dirname(os.path.realpath(__file__))+'/'


from gimpfu import *
import sys
sys.path.extend([baseLoc+'gimpenv/lib/python2.7',baseLoc+'gimpenv/lib/python2.7/site-packages',baseLoc+'gimpenv/lib/python2.7/site-packages/setuptools'])


from PIL import Image
import torch
from torchvision import transforms, datasets
import numpy as np

def getSeg(input_image):
    model = torch.load(baseLoc+'deeplabv3/deeplabv3+model.pt')
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_image = Image.fromarray(input_image)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)


    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)

    tmp = np.array(r)
    tmp2 = 10*np.repeat(tmp[:, :, np.newaxis], 3, axis=2)


    return  tmp2


def channelData(layer):#convert gimp image to numpy
    region=layer.get_pixel_rgn(0, 0, layer.width,layer.height)
    pixChars=region[:,:] # Take whole layer
    bpp=region.bpp
    return np.frombuffer(pixChars,dtype=np.uint8).reshape(layer.height,layer.width,bpp)

def createResultLayer(image,name,result):
    rlBytes=np.uint8(result).tobytes();
    rl=gimp.Layer(image,name,image.width,image.height,image.active_layer.type,100,NORMAL_MODE)
    region=rl.get_pixel_rgn(0, 0, rl.width,rl.height,True)
    region[:,:]=rlBytes
    image.add_layer(rl,0)
    gimp.displays_flush()

def deeplabv3(img, layer) :
    if torch.cuda.is_available():
        gimp.progress_init("(Using GPU) Generating semantic segmentation map for " + layer.name + "...")
    else:
        gimp.progress_init("(Using CPU) Generating semantic segmentation map for " + layer.name + "...")

    imgmat = channelData(layer) 
    cpy=getSeg(imgmat)    
    createResultLayer(img,'new_output',cpy)


register(
    "deeplabv3",
    "deeplabv3",
    "Generate semantic segmentation map based on deep learning.",
    "Kritik Soman",
    "GIMP-ML",
    "2020",
    "deeplabv3...",
    "*",      # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [   (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None)
    ],
    [],
    deeplabv3, menu="<Image>/Layer/GIML-ML")

main()
