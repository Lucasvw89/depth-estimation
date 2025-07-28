# class DepthEstimator(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

#         self.encode1 = nn.Sequential(
#             # double conv
#             nn.Conv2d(
#                 in_channels=3, out_channels=16,
#                 kernel_size=3, padding=1
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=16, out_channels=16,
#                 kernel_size=3, padding=1
#             ),
#             nn.ReLU(),
#             # nn.BatchNorm2d(64),
#         )

#         self.encode2 = nn.Sequential(
#             # double conv
#             nn.Conv2d(
#                 in_channels=16, out_channels=64,
#                 kernel_size=3, padding=1
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=64, out_channels=64,
#                 kernel_size=3, padding=1
#             ),
#             nn.ReLU(),
#             # nn.BatchNorm2d(256),
#         )

#         self.encode3 = nn.Sequential(
#             # double conv
#             nn.Conv2d(
#                 in_channels=64, out_channels=256,
#                 kernel_size=3, padding=1
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=256, out_channels=256,
#                 kernel_size=3, padding=1
#             ),
#             nn.ReLU(),
#             # nn.BatchNorm2d(1024),
#         )

#         self.decode1 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=256, out_channels=64,
#                 kernel_size=3, padding=1
#             ),
#             nn.ReLU(),
#         )

#         self.decode2 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=128, out_channels=16,
#                 kernel_size=3, padding=1
#             ),
#             nn.ReLU(),
#         )

#         self.decode3 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=32, out_channels=4,
#                 kernel_size=3, padding=1
#             ),
#             nn.ReLU(),
#         )

#         self.conv1 = nn.Conv2d(4, 1, 1)

#     def forward(self, x):
#         x1 = self.encode1(x)
#         x1 = self.maxpool(x1)
#         x2 = self.encode2(x1)
#         x2 = self.maxpool(x2)
#         x3 = self.encode3(x2)

#         d1 = self.decode1(x3)
#         d2 = self.decode2(torch.cat([x2, d1], dim=1))
#         d2 = self.upsample(d2)
#         d3 = self.decode3(torch.cat([x1, d2], dim=1))
#         d3 = self.upsample(d3)
#         return self.conv1(d3)