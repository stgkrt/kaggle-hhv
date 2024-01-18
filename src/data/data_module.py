import os
import sys

import pytorch_lightning as L
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import ExpConfig
from data._dataloader import SegDataset


class DataModule(L.LightningDataModule):
    def __init__(self, config: ExpConfig) -> None:
        super().__init__()
        self.config = config

    def train_dataloader(self) -> DataLoader:
        df = pd.read_csv(self.config.train_df)
        df = df[df["data_name"].isin(self.config.train_data_name)].reset_index(
            drop=True
        )
        if len(df) == 0:
            print("No training data. df path:", self.config.train_df)
            raise RuntimeError
        dataset = SegDataset(df, self.config, phase="train")
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        df = pd.read_csv(self.config.valid_df)
        df = df[df["data_name"].isin(self.config.valid_data_name)].reset_index(
            drop=True
        )
        if len(df) == 0:
            print("No validation data. df path:", self.config.valid_df)
            raise RuntimeError
        dataset = SegDataset(df, self.config, phase="valid")
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
