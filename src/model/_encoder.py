import timm
import torch
import torch.nn as nn

from src.conf import ExpConfig


class SimpleTimmEncoder(nn.Module):
    """
    timm based encoder(check timm.list_models() for available models)
    """

    def __init__(self, config: ExpConfig):
        super().__init__()
        self.encoder = timm.create_model(
            config.encoder_name,
            pretrained=config.pretrained,
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

    x = torch.randn(1, 3, 256, 256)
    skip_connections = model(x)
    for idx, sk in enumerate(skip_connections):
        print(idx, "skip connection", sk.shape)

    print("feature channels list", model.get_encoder_channels())
