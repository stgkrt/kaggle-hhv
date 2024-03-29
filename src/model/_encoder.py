import os
import sys

import timm
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import ExpConfig


class SimpleTimmEncoder(nn.Module):
    """
    timm based encoder(check timm.list_models() for available models)
    """

    def __init__(self, config: ExpConfig, phase: str = "train"):
        super().__init__()
        if phase == "train":
            pretrained = True
        else:
            pretrained = False
        self.encoder = timm.create_model(
            config.encoder_name,
            pretrained=pretrained,
            in_chans=config.in_channels,
            features_only=True,
        )

    def get_encoder_channels(self):
        """get encoder channels"""
        return self.encoder.feature_info.channels()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = self.encoder(x)
        return skip_connections


if __name__ == "__main__":
    config = ExpConfig()
    model = SimpleTimmEncoder(config)

    x = torch.randn(1, config.in_channels, config.img_height, config.img_width)
    skip_connections = model(x)
    for idx, sk in enumerate(skip_connections):
        print(idx, "skip connection", sk.shape)

    print("feature channels list", model.get_encoder_channels())
