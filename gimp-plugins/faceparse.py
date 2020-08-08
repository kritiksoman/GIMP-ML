import os 
baseLoc = os.path.dirname(os.path.realpath(__file__))+'/'


from gimpfu import *
import sys

sys.path.extend([baseLoc+'gimpenv/lib/python2.7',baseLoc+'gimpenv/lib/python2.7/site-packages',baseLoc+'gimpenv/lib/python2.7/site-packages/setuptools',baseLoc+'face-parsing.PyTorch'])


from model import BiSeNet
from PIL import Image
import torch
from torchvision import transforms, datasets
import numpy as np

colors = np.array([[0,0,0],
[204,0,0],
[0,255,255],
[51,255,255],
[51,51,255],
[204,0,204],
[204,204,0],
[102,51,0],
[255,0,0],
[0,204,204],
[76,153,0],
[102,204,0],
[255,255,0],
[0,0,153],
[255,153,51],
[0,51,0],
[0,204,0],
[0,0,204],
[255,51,153]])
colors = colors.astype(np.uint8)

def getlabelmat(mask,idx):
    x=np.zeros((mask.shape[0],mask.shape[1],3))
    x[mask==idx,0]=colors[idx][0] 
    x[mask==idx,1]=colors[idx][1] 
    x[mask==idx,2]=colors[idx][2]
    return x 

def colorMask(mask):
    x=np.zeros((mask.shape[0],mask.shape[1],3))
    for idx in range(19):
        x=x+getlabelmat(mask,idx)
    return np.uint8(x)

def getface(input_image):
    save_pth = baseLoc+'face-parsing.PyTorch/79999_iter.pth'
    input_image = Image.fromarray(input_image)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    if torch.cuda.is_available():
        net.cuda()
        net.load_state_dict(torch.load(save_pth))
    else:
        net.load_state_dict(torch.load(save_pth, map_location=lambda storage, loc: storage))


    net.eval()

    
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    with torch.no_grad():
        img = input_image.resize((512, 512), Image.BILINEAR)
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        if torch.cuda.is_available():
            img = img.cuda()
        out = net(img)[0]
        if torch.cuda.is_available():
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        else:
            parsing = out.squeeze(0).numpy().argmax(0)
    
    parsing = Image.fromarray(np.uint8(parsing))
    parsing = parsing.resize(input_image.size) 
    parsing = np.array(parsing)

    return parsing

def getSeg(input_image):
    model = torch.load(baseLoc+'deeplabv3+model.pt')
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_image = Image.fromarray(input_image)

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
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
    rl=gimp.Layer(image,name,image.width,image.height,0,100,NORMAL_MODE)
    region=rl.get_pixel_rgn(0, 0, rl.width,rl.height,True)
    region[:,:]=rlBytes
    image.add_layer(rl,0)
    gimp.displays_flush()

def faceparse(img, layer) :
    if torch.cuda.is_available():
        gimp.progress_init("(Using GPU) Running face parse for " + layer.name + "...")
    else:
        gimp.progress_init("(Using CPU) Running face parse for " + layer.name + "...")

    imgmat = channelData(layer)
    if imgmat.shape[2] == 4:  # get rid of alpha channel
        imgmat = imgmat[:,:,0:3]
    cpy=getface(imgmat)
    cpy = colorMask(cpy)
    createResultLayer(img,'new_output',cpy)


    

register(
    "faceparse",
    "faceparse",
    "Running face parse.",
    "Kritik Soman",
    "Your",
    "2020",
    "faceparse...",
    "*",      # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [   (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
    ],
    [],
    faceparse, menu="<Image>/Layer/GIML-ML")

main()
