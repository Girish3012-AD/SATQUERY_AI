from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two convolution layers with normalization and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Max-pooling followed by convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling followed by feature fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class BuildingUNet(nn.Module):
    """
    Lightweight 4-band building segmentation network.

    Input:
        [B, 4, H, W]

    Output:
        [B, 1, H, W]

    Output values are logits. Apply sigmoid for probabilities.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels must be positive."
            )

        if base_channels <= 0:
            raise ValueError(
                "base_channels must be positive."
            )

        self.in_channels = in_channels
        self.base_channels = base_channels

        self.input_block = DoubleConv(
            in_channels,
            base_channels,
        )

        self.down1 = DownBlock(
            base_channels,
            base_channels * 2,
        )

        self.down2 = DownBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.bottleneck = DownBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.up1 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
        )

        self.up2 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
        )

        self.up3 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels,
        )

        self.output = nn.Conv2d(
            base_channels,
            1,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Expected input shape [B,C,H,W], "
                f"got {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {x.shape[1]}"
            )

        x1 = self.input_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.bottleneck(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.output(x)
