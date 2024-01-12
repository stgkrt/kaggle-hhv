import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from conf import ExpConfig


class SegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: ExpConfig):
        self.df = df
        self.config = config
        self.processed_data_dir = f"{self.config.processed_data_dir}"
        self.processed_data_dir += f"_{config.stride_height}_{config.stride_width}"
        self.mean_dict = {
            "kidney_1_dense": 91.522667,
            "kidney_2": 131.317631,
            "kidney_3_dense": 76.838148,
        }
        self.std_dict = {
            "kidney_1_dense": 11.151333,
            "kidney_2": 9.339552,
            "kidney_3_dense": 2.465148,
        }

    def __len__(self) -> int:
        return len(self.df)

    def _normalize_img(self, image: np.ndarray, mean: float, std: float) -> np.ndarray:
        image = image.astype(np.float32)
        # mean = image.mean()
        # std = image.std()
        # if std != 0:
        #     image = (image - mean) / std
        image = (image - mean) / std
        return image

    def _get_random_crop_img(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, h, w = image.shape  # channel, height, width
        img_height = self.config.img_height
        img_width = self.config.img_width
        if h <= img_height and w <= img_width:
            return image, mask
        else:
            h_offset = np.random.randint(0, h - img_height)
            w_offset = np.random.randint(0, w - img_width)
            crop_img = image[
                :, h_offset : h_offset + img_height, w_offset : w_offset + img_width
            ]
            crop_mask = mask[
                :, h_offset : h_offset + img_height, w_offset : w_offset + img_width
            ]
            return crop_img, crop_mask

    def _load_image(self, data_name: str, file_name: str, phase: str) -> torch.Tensor:
        image_file_path = os.path.join(
            self.processed_data_dir, phase, data_name, "images", file_name
        )
        image = np.load(image_file_path)
        image = self._normalize_img(
            image, self.mean_dict[data_name], self.std_dict[data_name]
        )
        image = np.expand_dims(image.astype(np.float32), axis=-1)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def _load_label(self, data_name: str, file_name: str, phase: str) -> torch.Tensor:
        mask_file_path = os.path.join(
            self.processed_data_dir, phase, data_name, "labels", file_name
        )
        mask = np.load(mask_file_path)
        mask = np.expand_dims(mask.astype(np.float32), axis=-1)
        mask = torch.from_numpy(mask > 0).permute(2, 0, 1).float()

        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_name = self.df.loc[idx, "data_name"]
        phase = self.df.loc[idx, "phase"]
        input = torch.empty(0)
        mask = torch.empty(0)
        for slice_idx in range(self.config.slice_num):
            file_name = self.df.loc[idx, f"file_name_{slice_idx}"]
            slice_img = self._load_image(data_name, file_name, phase)
            slice_mask = self._load_label(data_name, file_name, phase)
            slice_img, slice_mask = self._get_random_crop_img(slice_img, slice_mask)
            input = torch.cat([input, slice_img], dim=0)
            mask = torch.cat([mask, slice_mask], dim=0)
        return input, mask


if __name__ == "__main__":
    config = ExpConfig()
    train_df_path = os.path.join(
        config.output_dir,
        f"{config.phase}_{config.stride_height}_{config.stride_width}_debug.csv",
    )
    print(train_df_path)
    df = pd.read_csv(train_df_path)
    dataset = SegDataset(df, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    print("phase", config.phase)
    for idx, (image, mask) in enumerate(dataloader):
        print(idx)
        print(image.shape)
        print(mask.shape)
        if idx > 3:
            break

    print("===")
    config.phase = "valid"
    df = pd.read_csv(
        os.path.join(
            config.output_dir,
            f"{config.phase}_{config.stride_height}_{config.stride_width}_debug.csv",
        )
    )
    dataset = SegDataset(df, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    print("phase", config.phase)
    for idx, (image, mask) in enumerate(dataloader):
        print(idx)
        print(image.shape)
        print(mask.shape)
        if idx > 3:
            break
