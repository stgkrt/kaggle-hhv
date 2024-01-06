import os

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from src.conf import ExpConfig
from src.data._dataloader import SegDataset


class DataModule(L.LightningDataModule):
    def __init__(self, config: ExpConfig) -> None:
        super().__init__()
        self.config = config

    def train_dataloader(self) -> DataLoader:
        if self.config.debug:
            df = pd.read_csv(os.path.join(self.config.output_dir, "train_debug.csv"))
        else:
            df = pd.read_csv(os.path.join(self.config.output_dir, "train.csv"))
        df = df[df["data_name"].isin(self.config.train_data_name)]
        dataset = SegDataset(df, self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        if self.config.debug:
            df = pd.read_csv(os.path.join(self.config.output_dir, "valid_debug.csv"))
        else:
            df = pd.read_csv(os.path.join(self.config.output_dir, "valid.csv"))
        df = df[df["data_name"].isin(self.config.valid_data_name)]
        dataset = SegDataset(df, self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        return dataloader


if __name__ == "__main__":
    config = ExpConfig()
    dm = DataModule(config)
    train_loader = dm.train_dataloader()
    print("train_loader", len(train_loader))
    for batch in train_loader:
        print(batch[0].shape)
        print(batch[1].shape)
        break
    valid_loader = dm.val_dataloader()
    print("valid_loader", len(valid_loader))
    for batch in valid_loader:
        print(batch[0].shape)
        print(batch[1].shape)
        break
