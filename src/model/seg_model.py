import argparse

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from _decoder import SimpleUnetDecoder
from _encoder import SimpleTimmEncoder


class SegModel(nn.Module):
    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.encoder = SimpleTimmEncoder(config)
        config.encoder_channels = self.encoder.get_encoder_channels()
        config.decoder_channels = config.encoder_channels[::-1]
        self.decoder = SimpleUnetDecoder(config)
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.out_channels,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.segmentation_head(x)
        return x


if __name__ == "__main__":
    config = argparse.Namespace()
    config.encoder_name = "resnet18"
    config.pretrained = True
    config.in_channels = 3
    config.out_channels = 1
    config.use_batchnorm = True
    config.dropout = 0.2
    model = SegModel(config)

    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("output shape", y.shape)
