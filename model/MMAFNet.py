import torch
import torch.nn as nn
from utils.ops import conv_new, upsample_pixelshuffle, CAB
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, bias=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(channel, channel // reduction, 3, padding=1, bias=bias),
                                nn.ReLU(),
                                nn.Conv2d(channel // reduction, channel, 3, padding=1, bias=bias))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class channel_select(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, bias=False, padding=1, negative_slope=0.2):
        super(channel_select, self).__init__()
        self.in_c = in_c
        act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.BN_path = conv_new(in_c, in_c, kernel_size, stride, padding, bias, 'BLC', negative_slope)
        self.IN_path = conv_new(in_c, in_c, kernel_size, stride, padding, bias, 'ILC', negative_slope)
        self.weight = ChannelAttention(in_c * 2, 16)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cab = CAB(in_c, kernel_size, 8, False, act)
        self.conv = conv_new(in_c, out_c, kernel_size, stride, padding, bias, 'CL', negative_slope)

    def forward(self, in_feature):
        bn_feat = self.BN_path(in_feature)
        in_feat = self.IN_path(in_feature)
        combing_feat = torch.cat((bn_feat, in_feat), dim=1)
        combing_CA = self.weight(combing_feat)
        wight_CA = combing_CA.view(-1, 2, self.in_c)[:, :, :, None, None]
        CA = wight_CA[:, 0, ::] * bn_feat + wight_CA[:, 1, ::] * in_feat
        pool = self.pool(CA)
        cab = self.cab(pool)
        out_feat = self.conv(cab)
        return out_feat


class high_frequency(nn.Module):
    def __init__(self, feat_n, negative_slope=0.2):
        super(high_frequency, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.e_conv1 = nn.Conv2d(feat_n, feat_n, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(2 * feat_n, feat_n, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(3 * feat_n, feat_n, 3, 1, 1, bias=True)

    def forward(self, in_feat):
        x1 = self.relu(self.e_conv1(in_feat))
        x2 = self.relu(self.e_conv2(torch.cat((in_feat, x1), dim=1)))
        out = self.relu(self.e_conv3(torch.cat((in_feat, x1, x2), dim=1)))

        return out


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class drop_component(nn.Module):
    def __init__(self, in_c, out_c):
        super(drop_component, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, stride=1, kernel_size=3, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.weight = nn.Sequential(nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1, stride=1,
                                              padding=0, bias=False), nn.Sigmoid())

    def forward(self, cur_feat):
        conv = self.conv(cur_feat)
        weight = self.weight(cur_feat)
        return self.act(conv * weight + cur_feat)


class enhance_net_nopool(nn.Module):
    def __init__(self, n_feat):
        super(enhance_net_nopool, self).__init__()
        self.n_feat = n_feat
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(n_feat * 2, n_feat, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(n_feat * 3, n_feat, 3, 1, 1, bias=True)

    def forward(self, in_feat):
        r1 = self.relu(self.e_conv1(in_feat))
        r2 = self.relu(self.e_conv2(torch.cat((in_feat, r1), dim=1)))
        r3 = self.relu(self.e_conv3(torch.cat((in_feat, r1, r2), dim=1)))

        in_feat = in_feat + r1 * (torch.pow(in_feat, 2) - in_feat)
        in_feat = in_feat + r2 * (torch.pow(in_feat, 2) - in_feat)
        out_feat = in_feat + r3 * (torch.pow(in_feat, 2) - in_feat)

        return out_feat


class texture_color(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, feat_n=64, kernel_size=3, stride=1, padding=1, bias=False,
                 mode='CL', negative_slope=0.2):
        super(texture_color, self).__init__()
        self.d_nc = feat_n // 4 * 3
        self.r_nc = feat_n // 4
        self.in_channel = in_channels
        self.upsample = F.interpolate
        act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.install = conv_new(in_channels, feat_n, kernel_size, stride, padding, bias, 'CL', negative_slope)
        self.channel_select1 = channel_select(self.r_nc, feat_n)
        self.progressive1 = drop_component(self.d_nc, self.d_nc)

        self.channel_select2 = channel_select(self.r_nc, feat_n)
        self.progressive2 = drop_component(self.d_nc, self.d_nc)


        self.progressive3 = drop_component(self.d_nc, self.d_nc)
        self.enhance_net_nopool = enhance_net_nopool(self.r_nc)

        self.cab1 = CAB(feat_n, kernel_size, 8, False, act)
        self.cab2 = CAB(feat_n, kernel_size, 8, False, act)
        self.cab3 = CAB(feat_n, kernel_size, 8, False, act)

        self.high_frequency1 = high_frequency(self.r_nc)
        self.high_frequency2 = high_frequency(self.r_nc)

        self.up1 = nn.Sequential(
            conv_new(feat_n, self.r_nc, 1, stride, 0, bias, mode=mode, negative_slope=negative_slope), Scale())
        self.up2 = nn.Sequential(
            conv_new(feat_n, self.r_nc, 1, stride, 0, bias, mode=mode, negative_slope=negative_slope), Scale())

        self.tail = upsample_pixelshuffle(feat_n, out_channels, mode=str(2))

    def forward(self, input_feat):
        install = self.install(input_feat)
        d1, r1 = torch.split(install, (self.d_nc, self.r_nc), dim=1)
        ext_d1, drop_r1 = self.progressive1(d1), self.channel_select1(r1)
        d2, r2 = torch.split(drop_r1, (self.d_nc, self.r_nc), dim=1)
        ext_d2, drop_r2 = self.progressive2(d2), self.channel_select2(r2)
        d3, r3 = torch.split(drop_r2, (self.d_nc, self.r_nc), dim=1)
        ext_d3, enhance_net_nopool = self.progressive3(d3), self.enhance_net_nopool(r3)

        combing3 = self.cab3(torch.cat((ext_d3, enhance_net_nopool), dim=1))
        up2 = self.up2(self.upsample(combing3, size=r2.size()[2:4], mode='bilinear', align_corners=False))
        up_r2 = self.high_frequency2(r2 + up2)

        combing2 = self.cab2(torch.cat((ext_d2, up_r2), dim=1))
        up1 = self.up1(self.upsample(combing2, size=r1.size()[2:4], mode='bilinear', align_corners=False))
        up_r1 = self.high_frequency1(r1 + up1)

        combing1 = self.cab1(torch.cat((ext_d1, up_r1),dim=1))

        out = install + combing1
        tail = self.tail(out)
        return tail


class MMAFNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=64):
        super(MMAFNet, self).__init__()
        ##-------------------define class---------------------------------
        ##-------------------------begin X4 images-------------------------------------
        self.feat_extract1 = texture_color(in_channels=in_c, out_channels=n_feat)
        self.feat_extract2 = texture_color(in_channels=n_feat, out_channels=out_c)
        self.skip = upsample_pixelshuffle(in_c, out_c, mode=str(4))

    def forward(self, RGBImagesX4):
        ##-------------------------begin X4 images-------------------------------------
        out_feat1 = self.feat_extract1(RGBImagesX4)
        out_feat2 = self.feat_extract2(out_feat1)
        skip = self.skip(RGBImagesX4)
        tail = torch.clamp(out_feat2 + skip, -1, 1)
        return tail


if __name__ == '__main__':
    Y = torch.randn((2, 1, 128, 128))
    img = torch.randn((2, 3, 126, 128))
    netG = MMAFNet()
    netG(img)
    print(netG.parameters())
    print(sum(param.numel() for param in netG.parameters()))
