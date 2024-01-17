import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from conf import ExpConfig
from data._augmentations import get_transforms


def min_max_normalize_img(image: np.ndarray, min: float, max: float) -> np.ndarray:
    image = image.astype(np.float32)
    image = (image - min) / (max - min)
    return image


class SegDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, config: ExpConfig, phase: str = "train"
    ) -> None:
        self.df = df
        self.config = config
        self.processed_data_dir = f"{self.config.processed_data_dir}"
        self.processed_data_dir += f"_{config.stride_height}_{config.stride_width}"
        self.transform = get_transforms(config, phase)
        self.maxmin_df = pd.read_csv(config.minmax_df_path)

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
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w, _ = image.shape  # height, width, ch
        img_height = self.config.img_height
        img_width = self.config.img_width
        if h <= img_height and w <= img_width:
            return image, mask
        else:
            h_offset = np.random.randint(0, h - img_height)
            w_offset = np.random.randint(0, w - img_width)
            crop_img = image[
                h_offset : h_offset + img_height, w_offset : w_offset + img_width
            ]
            crop_mask = mask[
                h_offset : h_offset + img_height, w_offset : w_offset + img_width
            ]
            return crop_img, crop_mask

    def _load_image(self, data_name: str, file_name: str, phase: str) -> np.ndarray:
        image_file_path = os.path.join(
            self.processed_data_dir, phase, data_name, "images", file_name
        )
        image = np.load(image_file_path)
        # image = self._normalize_img(
        #     image, self.mean_dict[data_name], self.std_dict[data_name]
        # # )
        max_value = self.maxmin_df[self.maxmin_df["data_name"] == data_name][
            "max"
        ].values[0]
        min_value = self.maxmin_df[self.maxmin_df["data_name"] == data_name][
            "min"
        ].values[0]
        image = min_max_normalize_img(image, min_value, max_value)
        image = np.expand_dims(image.astype(np.float32), axis=-1)
        return image

    def _load_label(self, data_name: str, file_name: str, phase: str) -> np.ndarray:
        mask_file_path = os.path.join(
            self.processed_data_dir, phase, data_name, "labels", file_name
        )
        mask = np.load(mask_file_path)
        mask = np.expand_dims(mask.astype(np.float32), axis=-1)

        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_name = self.df.loc[idx, "data_name"]
        phase = self.df.loc[idx, "phase"]
        input = np.empty(0)
        mask = np.empty(0)
        for slice_idx in range(self.config.slice_num):
            file_name = self.df.loc[idx, f"file_name_{slice_idx}"]
            slice_img = self._load_image(data_name, file_name, phase)
            slice_mask = self._load_label(data_name, file_name, phase)
            slice_img, slice_mask = self._get_random_crop_img(slice_img, slice_mask)
            if slice_idx == 0:
                input = slice_img
                mask = slice_mask
            else:
                input = np.concatenate([input, slice_img], axis=-1)
                mask = np.concatenate([mask, slice_mask], axis=-1)
        transformed = self.transform(image=input, mask=mask)
        input = transformed["image"]
        mask = transformed["mask"]
        # input = input.transpose(2, 0, 1)
        # mask = mask.transpose(2, 0, 1)
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
