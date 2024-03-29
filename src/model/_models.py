import os
import sys

import segmentation_models_pytorch as smp
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import ExpConfig
from model._decoder import SimpleUnetDecoder
from model._encoder import SimpleTimmEncoder


class SimpleSegModel(nn.Module):
    def __init__(self, config: ExpConfig, phase: str = "train"):
        super().__init__()
        self.encoder = SimpleTimmEncoder(config, phase)
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
    config = ExpConfig()
    model = SimpleSegModel(config)
    import torch

    input = torch.randn(
        config.batch_size, config.in_channels, config.img_height, config.img_width
    )
    output = model(input)
    print(output.shape)
