import os
import sys

import lightning as L
import torch
from torchmetrics import MeanMetric

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import ExpConfig
from losses import DiceLoss
from model._models import SimpleSegModel

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
        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = DiceLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def on_train_start(self) -> None:
        self.val_loss.reset()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.train_loss(loss)
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
            prog_bar=True,
        )
        self.val_loss(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train_loss_epoch", self.train_loss.compute())
        return

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss_epoch", self.val_loss.compute())
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.T_max, eta_min=self.config.eta_min
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    config = ExpConfig()
    model = ModelModule(config)
