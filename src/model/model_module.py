import lightning as L
import torch
import torch.nn as nn

from src.conf import ExpConfig
from src.model._models import SimpleSegModel

# MODEL_TYPE = Union[SimpleSegModel]


def get_model(config: ExpConfig) -> SimpleSegModel:
    if config.model_name == "SegModel":
        model = SimpleSegModel(config)
    else:
        raise NotImplementedError
    return model


class ModelModule(L.LightningModule):
    def __init__(self, config: ExpConfig) -> None:
        super().__init__()
        self.config = config
        self.model = get_model(config)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.T_max, eta_min=self.config.eta_min
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    config = ExpConfig()
    model = ModelModule(config)
