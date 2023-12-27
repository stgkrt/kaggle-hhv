import argparse

import timm
import torch
import torch.nn as nn


class SimpleTimmEncoder(nn.Module):
    def __init__(self, config: argparse.Namespace):
        super().__init__()
        self.encoder = timm.create_model(
            config.encoder_name,
            pretrained=config.pretrained,
            in_chans=config.in_channels,
            features_only=True,
        )

    def get_encoder_channels(self):
        return self.encoder.feature_info.channels()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = self.encoder(x)
        return skip_connections


if __name__ == "__main__":
    config = argparse.Namespace()
    config.encoder_name = "resnet18"
    config.pretrained = True
    config.in_channels = 3
    model = SimpleTimmEncoder(config)

    x = torch.randn(1, 3, 256, 256)
    skip_connections = model(x)
    for idx, sk in enumerate(skip_connections):
        print(idx, "skip connection", sk.shape)

    print("feature channels list", model.get_encoder_channels())
