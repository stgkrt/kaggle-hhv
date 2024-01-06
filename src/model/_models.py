import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.conf import ExpConfig
from src.model._decoder import SimpleUnetDecoder
from src.model._encoder import SimpleTimmEncoder


class SimpleSegModel(nn.Module):
    def __init__(self, config: ExpConfig):
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
    config = ExpConfig()
    model = SimpleSegModel(config)
    # 下ので読み出せるけどあってない？
    # weights = torch.load("/kaggle/working/debug/last.ckpt")["state_dict"]
    # model.load_state_dict(weights, strict=False)

    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("output shape", y.shape)
