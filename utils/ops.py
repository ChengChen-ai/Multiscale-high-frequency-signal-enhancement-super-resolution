import torch.nn as nn
import torch
from collections import OrderedDict
import math
import torch.nn.functional as F

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

'''
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''
def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def conv_new(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)

def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv_new(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1

##########################################################################
## RGB2YCbCr Block (CAB)
class RGB2YCbCr(nn.Module):
    def __init__(self, name):
        super(RGB2YCbCr, self).__init__()
        self.name = name

    def YCbCr(self, RGB_image, threshold=255, mean=0.5, div=0.5):

        R1 = RGB_image[:, 0:1, :, :]
        G1 = RGB_image[:, 1:2, :, :]
        B1 = RGB_image[:, 2:3, :, :]

        Y = 0.2568 * R1 + 0.504 * G1 + 0.1237 * B1 + (16/threshold-mean)/div + (0.257 + 0.564 + 0.098) * mean / div
        Cb = -0.148 * R1 - 0.291 * G1 + 0.439 * B1 + (128/threshold-mean)/div + (-0.148 - 0.291 + 0.439) * mean / div
        Cr = 0.439 * R1 - 0.368 * B1 - 0.071 * B1 + (128/threshold-mean)/div + (0.439 - 0.368 - 0.071) * mean / div

        ycbcr = torch.cat((Y, Cb, Cr), dim=1)

        return ycbcr

##########################################################################
## Conv Block (CAB)
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=(1,1), activation='lrelu', batch_norm=True, bias= False):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.batch_norm = batch_norm
        self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.relu = torch.nn.ReLU(True)
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.tanh = torch.nn.Tanh()
        self.plrelu = torch.nn.PReLU()

    def forward(self, x):
        if self.batch_norm:
            conv = self.conv(x)
            out = self.bn(conv)
        else:
            out = self.conv(x)

        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'prelu':
            return self.plrelu(out)
        elif self.activation == 'no_act':
            return out
##########################################################################
## Deconv Block (CAB)
class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, output_padding=1, activation='lrelu', batch_norm=True):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, output_padding ,bias=False)
        self.batch_norm = batch_norm
        self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.relu = torch.nn.ReLU(True)
        self.tanh = torch.nn.Tanh()
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.plrelu = torch.nn.PReLU()

    def forward(self, x, resize=None):
        if self.batch_norm:
            if resize:
                out = self.bn(self.deconv(x,output_size=resize))
            else:
                out = self.bn(self.deconv(x))
        else:
            if resize:
                out = self.deconv(x,output_size=resize)
            else:
                out = self.deconv(x)


        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'prelu':
            return self.plrelu(out)
        elif self.activation == 'no_act':
            return out

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y + x

class CALayer1(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer1, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer1(n_feat, reduction, bias=True)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
## Spatial Attention Block (CAB)
class SAB(nn.Module):
    def __init__(self, kernel_size=3, bias=True):
        super(SAB, self).__init__()
        self.conv1 = conv(2, 1, kernel_size, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        wight = torch.cat([avg_out, max_out], dim=1)
        wight = self.conv1(wight)
        return self.sigmoid(wight) * x


def pooling(in_feat):
    X2H = in_feat.size(2)
    X2W = in_feat.size(3)

    # in_feat = F.pad(in_feat, [X2W % 2, 0, X2H % 2, 0])
    # X2H = in_feat.size(2)
    # X2W = in_feat.size(3)

    fusion_X2ltop = in_feat[:, :, 0:int(X2H / 2), 0:int(X2W / 2)]
    fusion_X2rtop = in_feat[:, :, 0:int(X2H / 2), math.ceil(X2W / 2):X2W]
    fusion_X2lbot = in_feat[:, :, math.ceil(X2H / 2):X2H, 0:int(X2W / 2)]
    fusion_X2rbot = in_feat[:, :, math.ceil(X2H / 2):X2H, math.ceil(X2W / 2):X2W]

    return torch.cat((fusion_X2ltop, fusion_X2rtop, fusion_X2lbot, fusion_X2rbot), dim=1)

def upsample(in_feat):
    channel = in_feat.size(1)
    decoder_X2ltop, decoder_X2rtop, decoder_X2lbot, decoder_X2rbot = torch.split(in_feat, channel // 4, dim=1)

    top_feat = torch.cat((decoder_X2ltop, decoder_X2rtop), dim=3)
    bot_feat = torch.cat((decoder_X2lbot, decoder_X2rbot), dim=3)
    decoder_X2 = torch.cat((top_feat, bot_feat), dim=2)

    return decoder_X2