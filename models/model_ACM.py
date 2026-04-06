from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torchvision.models.resnet import BasicBlock


class AsymBiChaFuse(nn.Module):
    def __init__(self, channels: int = 64, reduction: int = 4) -> None:
        super().__init__()
        bottleneck_channels = channels // reduction
        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.Sigmoid(),
        )
        self.bottomup = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.Sigmoid(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True),
        )

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        if high.shape[-2:] != low.shape[-2:]:
            high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        topdown_weight = self.topdown(high)
        bottomup_weight = self.bottomup(low)
        fused = 2 * low * topdown_weight + 2 * high * bottomup_weight
        return self.post(fused)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FCNHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.9) -> None:
        super().__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ACM(nn.Module):
    """BasicIRSTD ACM/ASKCResUNet implementation for grayscale IRSTD."""

    def __init__(
        self,
        in_channels: int = 1,
        layers: list[int] | tuple[int, int, int] = (3, 3, 3),
        channels: list[int] | tuple[int, int, int, int] = (8, 16, 32, 64),
        fuse_mode: str = "AsymBi",
        tiny: bool = False,
        classes: int = 1,
        norm_layer: type[nn.Module] = BatchNorm2d,
        groups: int = 1,
        **_: object,
    ) -> None:
        super().__init__()
        self.tiny = tiny
        self._norm_layer = norm_layer
        self.groups = groups
        self.momentum = 0.9
        stem_width = int(channels[0])

        if tiny:
            self.stem = nn.Sequential(
                norm_layer(in_channels, self.momentum),
                nn.Conv2d(in_channels, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width * 2, momentum=self.momentum),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                norm_layer(in_channels, momentum=self.momentum),
                nn.Conv2d(in_channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width * 2, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(BasicBlock, layers[0], channels[1], channels[1], stride=1)
        self.layer2 = self._make_layer(BasicBlock, layers[1], channels[2], channels[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, layers[2], channels[3], channels[2], stride=2)

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=4, stride=2, padding=1)
        self.uplayer2 = self._make_layer(BasicBlock, layers[1], channels[2], channels[2], stride=1)
        self.fuse2 = self._make_fuse_layer(fuse_mode, channels[2])

        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=4, stride=2, padding=1)
        self.uplayer1 = self._make_layer(BasicBlock, layers[0], channels[1], channels[1], stride=1)
        self.fuse1 = self._make_fuse_layer(fuse_mode, channels[1])

        self.head = FCNHead(channels[1], classes, momentum=self.momentum)

    def _make_layer(
        self,
        block: type[BasicBlock],
        blocks: int,
        out_channels: int,
        in_channels: int,
        stride: int,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or out_channels != in_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels * block.expansion, momentum=self.momentum),
            )

        layers = [block(in_channels, out_channels, stride, downsample, self.groups, norm_layer=norm_layer)]
        current_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(current_channels, out_channels, self.groups, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _make_fuse_layer(self, fuse_mode: str, channels: int) -> nn.Module:
        if fuse_mode != "AsymBi":
            raise ValueError(f"Unknown fuse_mode: {fuse_mode}")
        return AsymBiChaFuse(channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape

        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        up2_in = self.deconv2(c3)
        if up2_in.shape[-2:] != c2.shape[-2:]:
            up2_in = F.interpolate(up2_in, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.uplayer2(self.fuse2(up2_in, c2))

        up1_in = self.deconv1(up2)
        if up1_in.shape[-2:] != c1.shape[-2:]:
            up1_in = F.interpolate(up1_in, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.uplayer1(self.fuse1(up1_in, c1))
        pred = self.head(up1)

        if self.tiny:
            return pred
        return F.interpolate(pred, size=(height, width), mode="bilinear", align_corners=False)


ASKCResUNet = ACM

__all__ = ["ACM", "ASKCResUNet"]
