import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, skip: bool = False) -> None:
        super(ResidualBlock, self).__init__()

        self.main_path: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.0),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.0),
        )
        self.skip: bool = skip
        self.shortcut: nn.Sequential = nn.Sequential()
        if skip:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.main_path(x)
        if self.skip:
            out += self.shortcut(x)
        return out


class ResNetModel(nn.Module):
    def __init__(self, num_classes: int = 1) -> None:
        super(ResNetModel, self).__init__()

        # First two residual blocks with stride 2
        self.first2: nn.Sequential = nn.Sequential(
            ResidualBlock(8, 64, stride=2, skip=False),
            ResidualBlock(64, 128, stride=2, skip=False),
        )

        # Last six residual blocks with stride 1
        self.last6: nn.Sequential = nn.Sequential(
            ResidualBlock(128, 192, stride=1, skip=True),
            ResidualBlock(192, 256, stride=1, skip=True),
            ResidualBlock(256, 320, stride=1, skip=True),
            ResidualBlock(320, 384, stride=1, skip=True),
            ResidualBlock(384, 448, stride=1, skip=True),
            ResidualBlock(448, 512, stride=1, skip=True),
        )

        self.classifier: nn.Sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.first2(x)
        out = self.last6(out)
        out = self.classifier(out)
        return out
