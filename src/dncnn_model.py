import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_channels=1, depth=17, features=64):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, in_channels, kernel_size=3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise_pred = self.net(x)
        denoised = x - noise_pred
        return denoised