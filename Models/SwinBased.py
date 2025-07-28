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
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.up = UpSample(in_feats, out_feats)
        self.double_conv1 = DoubleConv(in_feats, out_feats, kernel_size=1)
        self.double_conv3 = DoubleConv(in_feats, out_feats, kernel_size=3)
        self.double_conv5 = DoubleConv(in_feats, out_feats, kernel_size=5)

        self.final_conv = DoubleConv(3 * out_feats, out_feats, kernel_size=1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        
        x1 = self.double_conv1(x)
        x3 = self.double_conv3(x)
        x5 = self.double_conv5(x)

        x = torch.cat([x1, x3, x5], dim=1)
        x = self.final_conv(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = DoubleConv(768, 768)
        self.Dec0 = DecoderBlock(768, 384)
        self.Dec1 = DecoderBlock(384, 192)
        self.Dec2 = DecoderBlock(192, 96)
        self.final_up = UpSample(96, 48, scale_factor=2)
        self.last_skip = nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(48, 1, kernel_size=1, stride=1)

    def forward(self, features):
        before_swin = features[-1]
        skip0 = features[1].permute(0, 3, 1, 2)
        skip1 = features[3].permute(0, 3, 1, 2)
        skip2 = features[5].permute(0, 3, 1, 2)
        final = features[7].permute(0, 3, 1, 2)

        # for i in features:
        #     print(i.shape)

        x = self.initial_conv(final)
        del final

        x = self.Dec0(x, skip2)
        del skip2

        x = self.Dec1(x, skip1)
        del skip1

        x = self.Dec2(x, skip0)
        del skip0

        x = self.final_up(x)
        # print(before_swin.shape, x.shape)
        x = torch.cat([before_swin, x], dim=1)
        x = self.last_skip(x)
        
        x = self.final_conv(x)

        return x


class SwinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(48 // 4, 48),
            nn.LeakyReLU(0.2),

            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(48 // 4, 48),
            nn.LeakyReLU(0.2),
        )
        self.features = models.swin_v2_t(weights='DEFAULT').features

    def forward(self, x):
        convolved = self.conv(x)
        # print(convolved.shape)
        all_features = [x]
        for i in self.features:
            all_features.append(i(all_features[-1]))
        return all_features[1:] + [convolved]


class DepthEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = SwinEncoder()
        self.dec = Decoder()

    def forward(self, x):
        return self.dec(self.enc(x))