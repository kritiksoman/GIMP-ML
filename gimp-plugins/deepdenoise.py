import os

baseLoc = os.path.dirname(os.path.realpath(__file__)) + '/'

from gimpfu import *
import sys

sys.path.extend([baseLoc + 'gimpenv/lib/python2.7', baseLoc + 'gimpenv/lib/python2.7/site-packages',
                 baseLoc + 'gimpenv/lib/python2.7/site-packages/setuptools', baseLoc + 'PD-Denoising-pytorch'])


from denoiser import *
from argparse import Namespace

def clrImg(Img,cFlag):
    w, h, _ = Img.shape
    opt = Namespace(color=1, cond=1, delog='logsdc', ext_test_noise_level=None,
                    k=0, keep_ind=None, mode='MC', num_of_layers=20, out_dir='results_bc',
                    output_map=0, ps=2, ps_scale=2, real_n=1, refine=0, refine_opt=1,
                    rescale=1, scale=1, spat_n=0, test_data='real_night', test_data_gnd='Set12',
                    test_noise_level=None, wbin=512, zeroout=0)
    c = 1 if opt.color == 0 else 3
    net = DnCNN_c(channels=c, num_of_layers=opt.num_of_layers, num_of_est=2 * c)
    est_net = Estimation_direct(c, 2 * c)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids)
    model_est = nn.DataParallel(est_net, device_ids=device_ids)# Estimator Model
    if torch.cuda.is_available() and not cFlag:
        ckpt_est = torch.load(baseLoc+'weights/deepdenoise/est_net.pth')
        ckpt = torch.load(baseLoc+'weights/deepdenoise/net.pth')
        model = model.cuda()
        model_est = model_est.cuda()
    else:
        ckpt = torch.load(baseLoc+'weights/deepdenoise/net.pth',map_location=torch.device("cpu"))
        ckpt_est = torch.load(baseLoc+'weights/deepdenoise/est_net.pth',map_location=torch.device("cpu"))
    model.load_state_dict(ckpt)
    model.eval()
    model_est.load_state_dict(ckpt_est)
    model_est.eval()
    gimp.progress_update(float(0.005))
    gimp.displays_flush()    

    Img = Img[:, :, ::-1]  # change it to RGB
    Img = cv2.resize(Img, (0, 0), fx=opt.scale, fy=opt.scale)
    if opt.color == 0:
        Img = Img[:, :, 0]  # For gray images
        Img = np.expand_dims(Img, 2)
    pss = 1
    if opt.ps == 1:
        pss = decide_scale_factor(Img / 255., model_est, color=opt.color, thre=0.008, plot_flag=1, stopping=4,
                                  mark=opt.out_dir + '/' + file_name)[0]
        # print(pss)
        Img = pixelshuffle(Img, pss)
    elif opt.ps == 2:
        pss = opt.ps_scale

    merge_out = np.zeros([w, h, 3])
    wbin = opt.wbin
    i = 0
    idx=0
    t=(w*h)/(wbin*wbin)
    while i < w:
        i_end = min(i + wbin, w)
        j = 0
        while j < h:
            j_end = min(j + wbin, h)
            patch = Img[i:i_end, j:j_end, :]
            patch_merge_out_numpy = denoiser(patch, c, pss, model, model_est, opt)
            merge_out[i:i_end, j:j_end, :] = patch_merge_out_numpy
            j = j_end
            idx=idx+1
            gimp.progress_update(float(idx)/float(t))
            gimp.displays_flush()
        i = i_end


    return merge_out[:, :, ::-1]


def channelData(layer):  # convert gimp image to numpy
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    bpp = region.bpp
    # return np.frombuffer(pixChars,dtype=np.uint8).reshape(len(pixChars)/bpp,bpp)
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(layer.height, layer.width, bpp)


def createResultLayer(image, name, result):
    rlBytes = np.uint8(result).tobytes();
    rl = gimp.Layer(image, name, image.width, image.height, 0, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()


def deepdenoise(img, layer,cFlag):
    if torch.cuda.is_available() and not cFlag:
        gimp.progress_init("(Using GPU) Denoising " + layer.name + "...")
    else:
        gimp.progress_init("(Using CPU) Denoising " + layer.name + "...")
    imgmat = channelData(layer)
    if imgmat.shape[2] == 4:  # get rid of alpha channel
        imgmat = imgmat[:,:,0:3]
    cpy = clrImg(imgmat,cFlag)
    createResultLayer(img, 'new_output', cpy)


register(
    "deep-denoising",
    "deep-denoising",
    "Denoise image based on deep learning.",
    "Kritik Soman",
    "Your",
    "2020",
    "deep-denoising...",
    "*",  # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [(PF_IMAGE, "image", "Input image", None),
     (PF_DRAWABLE, "drawable", "Input drawable", None),
     (PF_BOOL, "fcpu", "Force CPU", False)
     ],
    [],
    deepdenoise, menu="<Image>/Layer/GIML-ML")

main()