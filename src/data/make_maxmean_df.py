import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def save_centerslice_maxmean_df(
    data_dir: str, save_dir: str, use_data_ratio: float = 0.1
) -> pd.DataFrame:
    data_path_list = glob(os.path.join(data_dir, "*"))
    dataname_list, mean_list, std_list = [], [], []
    max_list, min_list = [], []
    for data_path in data_path_list:
        print(data_path)
        data_name = os.path.split(data_path)[-1]
        print(data_name)
        image_list = sorted(glob(os.path.join(data_path, "images", "*.tif")))
        if len(image_list) == 0:
            continue
        center_idx = len(image_list) // 2
        stride_width = int(len(image_list) * use_data_ratio / 2)
        stack_img = None
        for idx in tqdm(
            range(center_idx - stride_width, center_idx + stride_width + 1)
        ):
            img_path = image_list[idx]
            img = cv2.imread(img_path)
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
    use_data_ratio = 0.1
    df = save_centerslice_maxmean_df(train_dir, save_dir, use_data_ratio)
    print(df)
