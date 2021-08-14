import torch
from torch import nn
from DFNet_core import (
    get_norm,
    get_activation,
    Conv2dSame,
    ConvTranspose2dSame,
    UpBlock,
    EncodeBlock,
    DecodeBlock,
)
from builtins import *


class RefinementNet(nn.Module):
    def __init__(
        self,
        c_img=19,
        c_mask=1,
        mode="nearest",
        norm="batch",
        act_en="relu",
        act_de="leaky_relu",
        en_ksize=[7, 5, 5, 3, 3, 3, 3, 3],
        de_ksize=[3] * 8,
    ):
        super(RefinementNet, self).__init__()

        c_in = c_img + c_mask

        self.en1 = EncodeBlock(c_in, 96, en_ksize[0], 2, None, None)
        self.en2 = EncodeBlock(
            96, 192, en_ksize[1], stride=2, normalization=norm, activation=act_en
        )
        self.en3 = EncodeBlock(
            192, 384, en_ksize[2], stride=2, normalization=norm, activation=act_en
        )
        self.en4 = EncodeBlock(
            384, 512, en_ksize[3], stride=2, normalization=norm, activation=act_en
        )
        self.en5 = EncodeBlock(
            512, 512, en_ksize[4], stride=2, normalization=norm, activation=act_en
        )
        self.en6 = EncodeBlock(
            512, 512, en_ksize[5], stride=2, normalization=norm, activation=act_en
        )
        self.en7 = EncodeBlock(
            512, 512, en_ksize[6], stride=2, normalization=norm, activation=act_en
        )
        self.en8 = EncodeBlock(
            512, 512, en_ksize[7], stride=2, normalization=norm, activation=act_en
        )

        self.de1 = DecodeBlock(
            512, 512, 512, mode, 3, scale=2, normalization=norm, activation=act_de
        )
        self.de2 = DecodeBlock(
            512, 512, 512, mode, 3, scale=2, normalization=norm, activation=act_de
        )
        self.de3 = DecodeBlock(
            512, 512, 512, mode, 3, scale=2, normalization=norm, activation=act_de
        )
        self.de4 = DecodeBlock(
            512, 512, 512, mode, 3, scale=2, normalization=norm, activation=act_de
        )
        self.de5 = DecodeBlock(
            512, 384, 384, mode, 3, scale=2, normalization=norm, activation=act_de
        )
        self.de6 = DecodeBlock(
            384, 192, 192, mode, 3, scale=2, normalization=norm, activation=act_de
        )
        self.de7 = DecodeBlock(
            192, 96, 96, mode, 3, scale=2, normalization=norm, activation=act_de
        )
        self.de8 = DecodeBlock(
            96, 20, 20, mode, 3, scale=2, normalization=norm, activation=act_de
        )

        self.last_conv = nn.Sequential(Conv2dSame(c_in, 3, 1, 1), nn.Sigmoid())

    def forward(self, img, mask):
        out = torch.cat([mask, img], dim=1)
        out_en = [out]

        out = self.en1(out)
        out_en.append(out)
        out = self.en2(out)
        out_en.append(out)
        out = self.en3(out)
        out_en.append(out)
        out = self.en4(out)
        out_en.append(out)
        out = self.en5(out)
        out_en.append(out)
        out = self.en6(out)
        out_en.append(out)
        out = self.en7(out)
        out_en.append(out)
        out = self.en8(out)
        out_en.append(out)

        out = self.de1(out, out_en[-0 - 2])
        out = self.de2(out, out_en[-1 - 2])
        out = self.de3(out, out_en[-2 - 2])
        out = self.de4(out, out_en[-3 - 2])
        out = self.de5(out, out_en[-4 - 2])
        out = self.de6(out, out_en[-5 - 2])
        out = self.de7(out, out_en[-6 - 2])
        out = self.de8(out, out_en[-7 - 2])

        output = self.last_conv(out)

        output = mask * output + (1 - mask) * img[:, :3]

        return output
