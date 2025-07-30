import torch
from torch import nn
import torchvision
from torchvision import models

class DoubleConv(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.GroupNorm(out_feats // 4, out_feats),
            nn.LeakyReLU(0.2),

            nn.Conv2d(out_feats, out_feats, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.GroupNorm(out_feats // 4, out_feats),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, c_in, c_out, scale_factor=2, mode='nearest'):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            # nn.ConvTranspose2d(c_in, c_in, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(c_out // 4, c_out),
            nn.LeakyReLU(0.2),
        )


    def forward(self, x):
        return self.upsample(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_feats, out_feats, upsample=True):
        super().__init__()
        self.upsample = upsample
        # if not self.upsample:    
            # print("sem upsample slkaaaaaaaaaaa")
        self.up = UpSample(in_feats, in_feats)
        self.double_conv1 = DoubleConv(in_feats, out_feats, kernel_size=1)
        self.double_conv3 = DoubleConv(in_feats, out_feats, kernel_size=3)
        self.double_conv5 = DoubleConv(in_feats, out_feats, kernel_size=5)

        self.final_conv = DoubleConv(3 * out_feats, out_feats, kernel_size=1)

    def forward(self, x, skip):
        x = torch.cat([skip, x], dim=1)

        if self.upsample:
            # print("dei upsample hihihi")
            x = self.up(x)
        # else:
            # print("sem upsample slk")
        
        x1 = self.double_conv1(x)
        x3 = self.double_conv3(x)
        x5 = self.double_conv5(x)

        x = torch.cat([x1, x3, x5], dim=1)
        x = self.final_conv(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_dec = nn.Sequential(
            DoubleConv(1280, 320),
            DoubleConv(320, 160),
            UpSample(160, 96),
        )
        self.all_dec = nn.ModuleList([
            DecoderBlock(192, 32),
            DecoderBlock(64, 24),
            DecoderBlock(48, 16),
            DecoderBlock(32, 32, upsample=False),
            DecoderBlock(64, 32, upsample=False),
        ])
        self.final_conv = nn.Sequential(
            DoubleConv(32, 16),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, features):
        features = features[::-1]
        # print("start ")

        x = self.initial_dec(features[0])
        # print("primeira conv")

        for skip, dec in zip(features[1:], self.all_dec):
            # print("convloka")
            # print(x.shape, skip.shape)
            x = dec(x, skip)

        x = self.final_conv(x)

        return x


class MobileEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = models.mobilenet_v2(weights='DEFAULT').features

    def forward(self, x):
        all_features = [x]
        for i in self.features:
            all_features.append(i(all_features[-1]))
        '''
        0: torch.Size([1, 32, 112, 112])
        1: torch.Size([1, 16, 112, 112])
        torch.Size([1, 24, 56, 56])
        3: torch.Size([1, 24, 56, 56])
        torch.Size([1, 32, 28, 28])
        torch.Size([1, 32, 28, 28])
        6: torch.Size([1, 32, 28, 28])
        torch.Size([1, 64, 14, 14])
        torch.Size([1, 64, 14, 14])
        torch.Size([1, 64, 14, 14])
        torch.Size([1, 64, 14, 14])
        torch.Size([1, 96, 14, 14])
        torch.Size([1, 96, 14, 14])
        13: torch.Size([1, 96, 14, 14])
        torch.Size([1, 160, 7, 7])
        torch.Size([1, 160, 7, 7])
        torch.Size([1, 160, 7, 7])
        torch.Size([1, 320, 7, 7])
        18: torch.Size([1, 1280, 7, 7])
        '''
        all_features = all_features[1:]
        # for i in all_features: print(i.shape)
        return [all_features[i] for i in (0, 1, 3, 6, 13, 18)]


class DepthEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = MobileEncoder()
        self.dec = Decoder()

    def forward(self, x):
        return self.dec(self.enc(x))