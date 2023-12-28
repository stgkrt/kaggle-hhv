"""
Modified the segmentation_model_pytorch U-Net decoder
https://github.com/qubvel/segmentation_models.pytorch
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: int = 0,
    ) -> None:
        super().__init__()

        conv_in_channels = in_channels + skip_channels

        # Convole input embedding and upscaled embedding
        self.conv1 = md.Conv2dReLU(
            conv_in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.dropout_skip = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if isinstance(skip, torch.Tensor):
            skip = self.dropout_skip(skip)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class SimpleUnetDecoder(nn.Module):
    def __init__(
        self,
        config: argparse.Namespace,
    ):
        super().__init__()
        encoder_channels = config.encoder_channels
        decoder_channels = config.decoder_channels
        use_batchnorm = config.use_batchnorm
        dropout = config.dropout

        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(encoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]

        self.center = nn.Identity()

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(
                in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm, dropout=dropout
            )
            for in_ch, skip_ch, out_ch in zip(
                in_channels, skip_channels, decoder_channels
            )
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


if __name__ == "__main__":
    config = argparse.Namespace()
    config.encoder_channels = [64, 64, 128, 256, 512]
    config.decoder_channels = [512, 256, 128, 64, 64]
    config.use_batchnorm = True
    config.dropout = 0
    model = SimpleUnetDecoder(config)

    skip_connection_shapes = [
        (1, 64, 128, 128),
        (1, 64, 64, 64),
        (1, 128, 32, 32),
        (1, 256, 16, 16),
        (1, 512, 8, 8),
    ]
    skip_connections = []
    for shape in skip_connection_shapes:
        skip_connections.append(torch.randn(shape))
    out = model(skip_connections)
    print(out.shape)