import torch
from torch import nn
import torchvision
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, c_in, c_out, scale_factor=2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=1),
        )


    def forward(self, x):
        return self.upsample(x)


class SwinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.swin_v2_t(weights='DEFAULT').features

    def forward(self, x):
        all_features = [x]
        for i in self.features:
            all_features.append(i(all_features[-1]))
        return all_features[1:]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = DoubleConv(768, 768)
        self.up0 = UpSample(768, 384)
        self.double_conv0 = DoubleConv(768, 384)
        self.up1 = UpSample(384, 192)
        self.double_conv1 = DoubleConv(384, 192)
        self.up2 = UpSample(192, 96)
        self.double_conv2 = DoubleConv(192, 96)
        self.final_up = UpSample(96, 96, scale_factor=4)
        self.final_conv = nn.Conv2d(96, 1, kernel_size=1, stride=1)

    def forward(self, features):
        skip0 = features[1].permute(0, 3, 1, 2)
        skip1 = features[3].permute(0, 3, 1, 2)
        skip2 = features[5].permute(0, 3, 1, 2)
        final = features[7].permute(0, 3, 1, 2)

        x = self.initial_conv(final)
        del final

        x = self.up0(x)
        x = torch.cat([skip2, x], dim=1)
        x = self.double_conv0(x)
        del skip2

        x = self.up1(x)
        x = torch.cat([skip1, x], dim=1)
        x = self.double_conv1(x)
        del skip1

        x = self.up2(x)
        x = torch.cat([skip0, x], dim=1)
        x = self.double_conv2(x)
        del skip0

        x = self.final_up(x)
        x = self.final_conv(x)

        return x


class DepthEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = SwinEncoder()
        self.dec = Decoder()

    def forward(self, x):
        return self.dec(self.enc(x))