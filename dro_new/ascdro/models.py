"""Model definitions used by ASCDRO experiments."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models as tv_models


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """ResNet backbone tailored for CIFAR inputs (32x32)."""

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: Sequence[int],
        num_classes: int = 10,
        width_multiplier: float = 1.0,
        stage_widths: Sequence[int] | None = None,
        extra_conv: bool = False,
        extra_conv_factor: float = 1.0,
        extra_conv_layers: int = 1,
    ) -> None:
        super().__init__()

        if stage_widths is not None and len(stage_widths) != 3:
            raise ValueError("stage_widths must contain exactly three entries for CIFAR stages.")

        base_channels = list(stage_widths) if stage_widths is not None else [32, 64, 128]
        channels = [max(16, int(round(c * width_multiplier))) for c in base_channels]

        self._num_blocks = tuple(num_blocks)
        self._width_multiplier = width_multiplier
        self._stage_widths = tuple(channels)
        self.depth = 6 * num_blocks[0] + 2  # standard CIFAR ResNet depth formula

        self.in_planes = channels[0]
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.extra_conv = None
        self._extra_conv_factor = extra_conv_factor
        self._extra_conv_layers = extra_conv_layers
        final_channels = channels[2] * block.expansion
        if extra_conv:
            extra_out = max(8, int(round(final_channels * extra_conv_factor)))
            layers = []
            in_channels = final_channels
            layers_needed = max(1, extra_conv_layers)
            for _ in range(layers_needed):
                layers.append(
                    nn.Conv2d(in_channels, extra_out, kernel_size=3, stride=1, padding=1, bias=False)
                )
                layers.append(nn.BatchNorm2d(extra_out))
                layers.append(nn.ReLU(inplace=True))
                in_channels = extra_out
            self.extra_conv = nn.Sequential(*layers)
            final_channels = extra_out

        self.fc = nn.Linear(final_channels, num_classes)

        self._init_weights()

    def _make_layer(self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.extra_conv is not None:
            out = self.extra_conv(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def build_resnet_cifar(
    depth: int = 20,
    width_multiplier: float = 1.0,
    num_classes: int = 10,
    extra_conv: bool = False,
    extra_conv_factor: float = 1.0,
    stage_widths: Sequence[int] | None = None,
    extra_conv_layers: int = 1,
) -> ResNetCIFAR:
    depth_to_blocks = {
        20: (3, 3, 3),
        32: (5, 5, 5),
        44: (7, 7, 7),
        56: (9, 9, 9),
        110: (18, 18, 18),
        164: (27, 27, 27),
        200: (33, 33, 33),
        290: (48, 48, 48),
    }
    if depth not in depth_to_blocks:
        raise ValueError(f"Unsupported ResNet depth {depth}. Choose from {sorted(depth_to_blocks.keys())}.")
    return ResNetCIFAR(
        BasicBlock,
        depth_to_blocks[depth],
        num_classes=num_classes,
        width_multiplier=width_multiplier,
        extra_conv=extra_conv,
        extra_conv_factor=extra_conv_factor,
        stage_widths=stage_widths,
        extra_conv_layers=extra_conv_layers,
    )


def resnet20(num_classes: int = 10) -> ResNetCIFAR:
    return build_resnet_cifar(depth=20, width_multiplier=1.0, num_classes=num_classes)


def build_resnet_imagenet(depth: int = 50, num_classes: int = 1000, pretrained: bool = False) -> nn.Module:
    """Build an ImageNet-style ResNet using torchvision implementations."""

    depth_to_builder = {
        18: tv_models.resnet18,
        34: tv_models.resnet34,
        50: tv_models.resnet50,
        101: tv_models.resnet101,
        152: tv_models.resnet152,
    }
    if depth not in depth_to_builder:
        raise ValueError(f"Unsupported ImageNet ResNet depth {depth}. Choose from {sorted(depth_to_builder.keys())}.")

    if pretrained:
        raise NotImplementedError("Pretrained weights are not supported in this configuration.")

    builder = depth_to_builder[depth]
    try:
        model = builder(weights=None)
    except TypeError:
        model = builder(pretrained=False)

    if num_classes != getattr(model.fc, "out_features", num_classes):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model
