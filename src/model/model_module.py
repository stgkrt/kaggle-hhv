import os
import sys
from typing import Dict, List

import cv2
import lightning as L
import numpy as np
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

    def _pad_img_ifneed(
        self, img: np.ndarray, img_height: int, img_width: int
    ) -> np.ndarray:
        """image size is smaller than img_height or img_width, pad image

        Args:
            img (np.ndarray): image [H, W, C] ndim = 2 or 3

        Returns:
            np.ndarray: padded image [img_height, img_width, C]
        """
        if (img.shape[0] < img_height) or (img.shape[1] < img_width):
            pad_size = (
                img_height - img.shape[0],
                img_width - img.shape[1],
            )
            img = np.pad(img, ((0, pad_size[0]), (0, pad_size[1])), "constant")
        return img

    def _init_batch_input_dict(self) -> dict:
        batch_input_dict: Dict[str, List[int]] = {
            "h_start": [],
            "w_start": [],
            "h_end": [],
            "w_end": [],
        }
        return batch_input_dict

    def _pred_batch(
        self,
        batch_input: torch.Tensor,
        batch_input_dict: dict,
        pred_img: np.ndarray,
        pred_count_img: np.ndarray,
    ) -> tuple[torch.Tensor, dict[str, List[int]], np.ndarray, np.ndarray]:
        batch_input = batch_input.to(self.device)
        preds = self.model(batch_input)
        for i in range(batch_input.shape[0]):
            h_start = batch_input_dict["h_start"][i]
            w_start = batch_input_dict["w_start"][i]
            h_end = batch_input_dict["h_end"][i]
            w_end = batch_input_dict["w_end"][i]
            pred = preds[i].squeeze().detach().cpu().numpy()
            pred = pred[: h_end - h_start, : w_end - w_start]
            pred_img[h_start:h_end, w_start:w_end] += pred
            pred_count_img[h_start:h_end, w_start:w_end] += np.ones_like(pred)
        batch_input = torch.empty(0)
        batch_input_dict = self._init_batch_input_dict()

        return batch_input, batch_input_dict, pred_img, pred_count_img

    def overlap_predict(
        self, original_img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """overlapで指定した割合で重なるように画像を分割して推論する

        Args:
            original_img (np.ndarray): original image [H, W, C]

        Returns:
            np.ndarray: predicted image [H, W, C]
        """
        overlap_rate = self.config.overlap_rate
        stride_height = int(self.config.img_height * overlap_rate)
        stride_width = int(self.config.img_width * overlap_rate)
        h_stride_num = int(
            np.ceil((original_img.shape[0] - self.config.img_height) / stride_height)
            + 1
        )
        w_stride_num = int(
            np.ceil((original_img.shape[1] - self.config.img_width) / stride_width) + 1
        )
        pred_img = np.zeros_like(original_img, dtype=np.float32)
        pred_count_img = np.zeros_like(original_img, dtype=np.float32)
        batch_input = torch.empty(0)
        batch_input_dict: Dict[str, List[int]] = {
            "h_start": [],
            "w_start": [],
            "h_end": [],
            "w_end": [],
        }
        for h_idx in range(h_stride_num):
            for w_idx in range(w_stride_num):
                h_start = h_idx * stride_height
                w_start = w_idx * stride_width
                h_end = min(h_start + self.config.img_height, original_img.shape[0])
                w_end = min(w_start + self.config.img_width, original_img.shape[1])
                crop_img = original_img[h_start:h_end, w_start:w_end]
                crop_img = self._pad_img_ifneed(
                    crop_img, self.config.img_height, self.config.img_width
                )
                crop_img = torch.from_numpy(crop_img).unsqueeze(0)  # [C, H, W]
                crop_img = crop_img.unsqueeze(0)  # [B, C, H, W]
                batch_input = torch.cat([batch_input, crop_img], dim=0)
                batch_input_dict["h_start"].append(h_start)
                batch_input_dict["w_start"].append(w_start)
                batch_input_dict["h_end"].append(h_end)
                batch_input_dict["w_end"].append(w_end)
                if batch_input.shape[0] == self.config.batch_size:
                    (
                        batch_input,
                        batch_input_dict,
                        pred_img,
                        pred_count_img,
                    ) = self._pred_batch(
                        batch_input, batch_input_dict, pred_img, pred_count_img
                    )
        (
            batch_input,
            batch_input_dict,
            pred_img,
            pred_count_img,
        ) = self._pred_batch(batch_input, batch_input_dict, pred_img, pred_count_img)
        pred_img = pred_img / pred_count_img
        pred_img = pred_img.astype(np.float32)
        return pred_img, pred_count_img


if __name__ == "__main__":
    config = ExpConfig()
    model = ModelModule(config)
    # これで読み出せる
    model.load_state_dict(torch.load("/kaggle/working/exp001_making3/last.pth"))
    # 1枚ずつ推論できる
    image = cv2.imread(
        "/kaggle/input/blood-vessel-segmentation/train/kidney_1_dense/images/0500.tif",
        cv2.IMREAD_GRAYSCALE,
    )
    pred, pred_count_img = model.overlap_predict(image)
