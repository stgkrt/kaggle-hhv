import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: argparse.Namespace):
        self.df = df
        self.config = config

    def __len__(self) -> int:
        return len(self.df)

    def _get_random_crop_img(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        img_height = self.config.img_height
        img_width = self.config.img_width
        h_offset = np.random.randint(0, h - img_height)
        w_offset = np.random.randint(0, w - img_width)
        crop_img = image[
            h_offset : h_offset + img_height, w_offset : w_offset + img_width
        ]
        return crop_img

    def _load_image(self, data_name: str, file_name: str, phase: str) -> torch.Tensor:
        image_file_path = os.path.join(
            self.config.processed_data_dir, phase, data_name, "images", file_name
        )
        image = np.load(image_file_path)
        image = np.expand_dims(image.astype(np.float32), axis=-1)
        image = self._get_random_crop_img(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def _load_label(self, data_name: str, file_name: str, phase: str) -> torch.Tensor:
        mask_file_path = os.path.join(
            self.config.processed_data_dir, phase, data_name, "labels", file_name
        )
        mask = np.load(mask_file_path)
        mask = np.expand_dims(mask.astype(np.float32), axis=-1)
        mask = self._get_random_crop_img(mask)
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        mask = (mask > 0.0).long()
        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_name = self.df.loc[idx, "data_name"]
        phase = self.df.loc[idx, "phase"]
        input = torch.empty(0)
        for slice_idx in range(self.config.slice_num):
            file_name = self.df.loc[idx, f"file_name_{slice_idx}"]
            one_slice = self._load_image(data_name, file_name, phase)
            input = torch.cat([input, one_slice], dim=0)
        mask = self._load_label(data_name, file_name, phase)
        return input, mask


if __name__ == "__main__":
    config = argparse.Namespace()
    config.input_dir = "/kaggle/input"
    config.competition_name = "blood-vessel-segmentation"
    config.input_data_dir = os.path.join(config.input_dir, config.competition_name)
    config.processed_data_dir = os.path.join("/kaggle", "working", "_processed")
    config.output_dir = "/kaggle/working"
    config.phase = "train"
    config.slice_num = 3
    config.img_height = 256
    config.img_width = 256

    df = pd.read_csv(os.path.join(config.output_dir, "train.csv"))
    print(df.head())
    dataset = SegDataset(df, config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for idx, (image, mask) in enumerate(dataloader):
        print(idx)
        print(image.shape)
        print(mask.shape)
        if idx > 10:
            break
