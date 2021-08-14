import torch
from torch import nn
import torch.nn.functional as F
from builtins import *


def resize_like(x, target, mode="bilinear"):
    return F.interpolate(x, target.shape[-2:], mode=mode, align_corners=False)


def get_norm(name, out_channels):
    if name == "batch":
        norm = nn.BatchNorm2d(out_channels)
    elif name == "instance":
        norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None
    return norm


def get_activation(name):
    if name == "relu":
        activation = nn.ReLU()
    elif name == "elu":
        activation == nn.ELU()
    elif name == "leaky_relu":
        activation = nn.LeakyReLU(negative_slope=0.2)
    elif name == "tanh":
        activation = nn.Tanh()
    elif name == "sigmoid":
        activation = nn.Sigmoid()
    else:
        activation = None
    return activation


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding = self.conv_same_pad(kernel_size, stride)
        if type(padding) is not tuple:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            )
        else:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding * 2, 0),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0),
            )

    def conv_same_pad(self, ksize, stride):
        if (ksize - stride) % 2 == 0:
            return (ksize - stride) // 2
        else:
            left = (ksize - stride) // 2
            right = left + 1
            return left, right

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding, output_padding = self.deconv_same_pad(kernel_size, stride)
        self.trans_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )

    def deconv_same_pad(self, ksize, stride):
        pad = (ksize - stride + 1) // 2
        outpad = 2 * pad + stride - ksize
        return pad, outpad

    def forward(self, x):
        return self.trans_conv(x)


class UpBlock(nn.Module):
    def __init__(self, mode="nearest", scale=2, channel=None, kernel_size=4):
        super().__init__()

        self.mode = mode
        if mode == "deconv":
            self.up = ConvTranspose2dSame(channel, channel, kernel_size, stride=scale)
        else:

            def upsample(x):
                return F.interpolate(x, scale_factor=scale, mode=mode)

            self.up = upsample

    def forward(self, x):
        return self.up(x)


class EncodeBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        normalization=None,
        activation=None,
    ):
        super().__init__()

        self.c_in = in_channels
        self.c_out = out_channels

        layers = []
        layers.append(Conv2dSame(self.c_in, self.c_out, kernel_size, stride))
        if normalization:
            layers.append(get_norm(normalization, self.c_out))
        if activation:
            layers.append(get_activation(activation))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class DecodeBlock(nn.Module):
    def __init__(
        self,
        c_from_up,
        c_from_down,
        c_out,
        mode="nearest",
        kernel_size=4,
        scale=2,
        normalization="batch",
        activation="relu",
    ):
        super().__init__()

        self.c_from_up = c_from_up
        self.c_from_down = c_from_down
        self.c_in = c_from_up + c_from_down
        self.c_out = c_out

        self.up = UpBlock(mode, scale, c_from_up, kernel_size=scale)

        layers = []
        layers.append(Conv2dSame(self.c_in, self.c_out, kernel_size, stride=1))
        if normalization:
            layers.append(get_norm(normalization, self.c_out))
        if activation:
            layers.append(get_activation(activation))
        self.decode = nn.Sequential(*layers)

    def forward(self, x, concat=None):
        out = self.up(x)
        if self.c_from_down > 0:
            out = torch.cat([out, concat], dim=1)
        out = self.decode(out)
        return out


class BlendBlock(nn.Module):
    def __init__(self, c_in, c_out, ksize_mid=3, norm="batch", act="leaky_relu"):
        super().__init__()
        c_mid = max(c_in // 2, 32)
        self.blend = nn.Sequential(
            Conv2dSame(c_in, c_mid, 1, 1),
            get_norm(norm, c_mid),
            get_activation(act),
            Conv2dSame(c_mid, c_out, ksize_mid, 1),
            get_norm(norm, c_out),
            get_activation(act),
            Conv2dSame(c_out, c_out, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.blend(x)


class FusionBlock(nn.Module):
    def __init__(self, c_feat, c_alpha=1):
        super().__init__()
        c_img = 3
        self.map2img = nn.Sequential(Conv2dSame(c_feat, c_img, 1, 1), nn.Sigmoid())
        self.blend = BlendBlock(c_img * 2, c_alpha)

    def forward(self, img_miss, feat_de):
        img_miss = resize_like(img_miss, feat_de)
        raw = self.map2img(feat_de)
        alpha = self.blend(torch.cat([img_miss, raw], dim=1))
        result = alpha * raw + (1 - alpha) * img_miss
        return result, alpha, raw


class DFNet(nn.Module):
    def __init__(
        self,
        c_img=3,
        c_mask=1,
        c_alpha=3,
        mode="nearest",
        norm="batch",
        act_en="relu",
        act_de="leaky_relu",
        en_ksize=[7, 5, 5, 3, 3, 3, 3, 3],
        de_ksize=[3] * 8,
        blend_layers=[0, 1, 2, 3, 4, 5],
    ):
        super().__init__()

        c_init = c_img + c_mask

        self.n_en = len(en_ksize)
        self.n_de = len(de_ksize)
        assert (
            self.n_en == self.n_de
        ), "The number layer of Encoder and Decoder must be equal."
        assert (
            self.n_en >= 1
        ), "The number layer of Encoder and Decoder must be greater than 1."

        assert 0 in blend_layers, "Layer 0 must be blended."

        self.en = []
        c_in = c_init
        self.en.append(EncodeBlock(c_in, 64, en_ksize[0], 2, None, None))
        for k_en in en_ksize[1:]:
            c_in = self.en[-1].c_out
            c_out = min(c_in * 2, 512)
            self.en.append(
                EncodeBlock(
                    c_in, c_out, k_en, stride=2, normalization=norm, activation=act_en
                )
            )

        # register parameters
        for i, en in enumerate(self.en):
            self.__setattr__("en_{}".format(i), en)

        self.de = []
        self.fuse = []
        for i, k_de in enumerate(de_ksize):

            c_from_up = self.en[-1].c_out if i == 0 else self.de[-1].c_out
            c_out = c_from_down = self.en[-i - 1].c_in
            layer_idx = self.n_de - i - 1

            self.de.append(
                DecodeBlock(
                    c_from_up,
                    c_from_down,
                    c_out,
                    mode,
                    k_de,
                    scale=2,
                    normalization=norm,
                    activation=act_de,
                )
            )
            if layer_idx in blend_layers:
                self.fuse.append(FusionBlock(c_out, c_alpha))
            else:
                self.fuse.append(None)

        # register parameters
        for i, de in enumerate(self.de[::-1]):
            self.__setattr__("de_{}".format(i), de)
        for i, fuse in enumerate(self.fuse[::-1]):
            if fuse:
                self.__setattr__("fuse_{}".format(i), fuse)

    def forward(self, img_miss, mask):

        out = torch.cat([img_miss, mask], dim=1)

        out_en = [out]
        for encode in self.en:
            out = encode(out)
            out_en.append(out)

        results = []
        alphas = []
        raws = []
        for i, (decode, fuse) in enumerate(zip(self.de, self.fuse)):
            out = decode(out, out_en[-i - 2])
            if fuse:
                result, alpha, raw = fuse(img_miss, out)
                results.append(result)

        return results[::-1]
