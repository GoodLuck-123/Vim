"""
Simple CNN baseline for depth estimation ablation study.

Can be used as a replacement for VisionMambaSeg to test data pipeline and training loop
without Mamba compilation overhead.
"""

import torch
import torch.nn as nn
from mmseg.models.builder import BACKBONES


@BACKBONES.register_module()
class CNNBaseline(nn.Module):
    """Simple CNN backbone for depth estimation.

    4-stage encoder with multi-scale features suitable for dense prediction tasks.
    Output: list of 4 feature maps at 1/4, 1/8, 1/16, 1/32 resolution.

    Args:
        in_chans (int): Number of input channels. Default: 3
        embed_dim (int): Base embedding dimension. Default: 64
        depths (tuple): Number of blocks per stage. Default: (2, 2, 6, 2)
    """

    def __init__(self, in_chans=3, embed_dim=64, depths=(2, 2, 6, 2)):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = depths

        # Stem: 1/2 resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 1/4 resolution
        self.layer1 = self._make_stage(embed_dim, embed_dim, depths[0], stride=2)

        # Stage 2: 1/8 resolution
        self.layer2 = self._make_stage(embed_dim, embed_dim * 2, depths[1], stride=2)

        # Stage 3: 1/16 resolution
        self.layer3 = self._make_stage(embed_dim * 2, embed_dim * 4, depths[2], stride=2)

        # Stage 4: 1/32 resolution
        self.layer4 = self._make_stage(embed_dim * 4, embed_dim * 6, depths[3], stride=2)

    def _make_stage(self, in_channels, out_channels, depth, stride):
        """Build a stage with multiple residual blocks."""
        layers = []

        # Downsample first block
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))

        # Remaining blocks
        for _ in range(depth - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass returning multi-scale features.

        Returns:
            list: 4 feature maps from stages 1-4
        """
        # Stem: input -> 1/2
        x = self.stem(x)

        # Stages with different resolutions
        x1 = self.layer1(x)    # 1/4
        x2 = self.layer2(x1)   # 1/8
        x3 = self.layer3(x2)   # 1/16
        x4 = self.layer4(x3)   # 1/32

        # Return multi-scale features for UPerHead decoder
        # Match VisionMambaSeg output: (1/4, 1/4, 1/4, 1/4) -> decoder upsample to image size
        return [x1, x2, x3, x4]


class ResidualBlock(nn.Module):
    """Simple residual block with bottleneck structure."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride

        hidden_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out
