import gc
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.processing.preprocess import clip_underover_value


def save_centerslice_maxmean_df(
    data_dir: str, save_dir: str, use_data_num: float = 100
) -> pd.DataFrame:
    data_path_list = glob(os.path.join(data_dir, "*"))
    dataname_list, mean_list, std_list = [], [], []
    max_list, min_list = [], []
    for data_path in sorted(data_path_list):
        print(data_path)
        data_name = os.path.split(data_path)[-1]
        print(data_name)
        image_list = sorted(glob(os.path.join(data_path, "images", "*.tif")))
        if len(image_list) == 0:
            continue
        if len(image_list) < use_data_num:
            use_data_num = len(image_list)
        stride = int(len(image_list) / use_data_num)
        print(f"image num: {len(image_list)}, stride: {stride}")
        stack_img = None
        for idx in tqdm(range(0, len(image_list), stride)):
            img_path = image_list[idx]
            img = cv2.imread(img_path)
            img_torch = torch.from_numpy(img).permute(2, 0, 1)
            img_torch = clip_underover_value(img_torch, percent=0.1)
            img = img_torch.numpy().astype(np.uint8)
            if stack_img is None:
                stack_img = np.expand_dims(img, axis=0)
            else:
                stack_img = np.concatenate(
                    [stack_img, np.expand_dims(img, axis=0)], axis=0
                )

        dataname_list.append(data_name)
        mean_list.append(np.mean(stack_img))
        std_list.append(np.std(stack_img))
        max_list.append(np.max(stack_img))
        min_list.append(np.min(stack_img))

        del stack_img
        gc.collect()
    df = pd.DataFrame(
        {
            "data_name": dataname_list,
            "mean": mean_list,
            "std": std_list,
            "max": max_list,
            "min": min_list,
        }
    )
    df.to_csv(os.path.join(save_dir, "centerslice_maxmean.csv"), index=False)

    return df


if __name__ == "__main__":
    train_dir = "/kaggle/input/blood-vessel-segmentation/train"
    save_dir = "/kaggle/working"
    use_data_num = 300
    df = save_centerslice_maxmean_df(train_dir, save_dir, use_data_num)
    print(df)
