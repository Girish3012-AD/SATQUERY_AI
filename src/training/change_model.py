from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class SharedEncoder(nn.Module):
    """
    Lightweight shared encoder.

    Both temporal images use exactly the same weights.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        self.enc1 = ConvBlock(
            in_channels,
            base_channels,
        )

        self.enc2 = ConvBlock(
            base_channels,
            base_channels * 2,
        )

        self.enc3 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.pool = nn.MaxPool2d(2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x1 = self.enc1(x)

        x2 = self.enc2(
            self.pool(x1)
        )

        x3 = self.enc3(
            self.pool(x2)
        )

        return x1, x2, x3


class ChangeDecoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = ConvBlock(
            base_channels * 4,
            base_channels * 2,
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = ConvBlock(
            base_channels * 2,
            base_channels,
        )

        self.head = nn.Conv2d(
            base_channels,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        difference: torch.Tensor,
        skip2: torch.Tensor,
        skip1: torch.Tensor,
    ) -> torch.Tensor:

        x = self.up2(difference)

        x = torch.cat(
            [x, skip2],
            dim=1,
        )

        x = self.dec2(x)

        x = self.up1(x)

        x = torch.cat(
            [x, skip1],
            dim=1,
        )

        x = self.dec1(x)

        return self.head(x)


class ChangeUNet(nn.Module):
    """
    Lightweight Siamese change-detection network.

    Input:
        T1: [B, 4, H, W]
        T2: [B, 4, H, W]

    Output:
        logits: [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        self.encoder = SharedEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
        )

        self.decoder = ChangeDecoder(
            base_channels=base_channels,
        )

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:

        before1, before2, before3 = self.encoder(before)

        after1, after2, after3 = self.encoder(after)

        # Absolute feature difference is symmetric with respect
        # to temporal ordering.
        difference = torch.abs(
            before3 - after3
        )

        return self.decoder(
            difference,
            torch.abs(before2 - after2),
            torch.abs(before1 - after1),
        )


if __name__ == "__main__":
    model = ChangeUNet()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "ChangeUNet parameters:",
        total_parameters,
    )

    before = torch.randn(
        2,
        4,
        64,
        64,
    )

    after = torch.randn(
        2,
        4,
        64,
        64,
    )

    with torch.no_grad():
        output = model(
            before,
            after,
        )

    print(
        "Before shape:",
        tuple(before.shape),
    )

    print(
        "After shape:",
        tuple(after.shape),
    )

    print(
        "Output shape:",
        tuple(output.shape),
    )

    print(
        "Output dtype:",
        output.dtype,
    )

    print(
        "Finite:",
        bool(torch.isfinite(output).all()),
    )
