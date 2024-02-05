import os
import sys
import time
from glob import glob
from typing import List

import cv2
import pandas as pd
import torch
from pytorch_lightning import LightningModule

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from data._dataloader import min_max_normalize_img
from processing.postprocess import remove_small_objects
from processing.preprocess import clip_underover_value
from score.rle_convert import rle_encode


def make_submit_df(
    model: LightningModule,
    data_dir: str,
    data_name_list: List[str],
    max_min_df_path: str | None = None,
    threshold: float = 0.5,
    object_min_size: int = 10,
) -> pd.DataFrame:
    data_id_list = []
    rle_list = []
    max_min_df = None
    if max_min_df_path is not None:
        max_min_df = pd.read_csv(max_min_df_path)
    for data_name in sorted(data_name_list):
        slice_id_list = sorted(
            [
                os.path.basename(path).split(".")[0]
                for path in glob(os.path.join(data_dir, data_name, "images", "*.tif"))
            ]
        )
        print(f"\n predicting... => {data_name}, slice num: {len(slice_id_list)}")
        if max_min_df is not None:
            data_values = max_min_df[max_min_df["data_name"] == data_name]
            max_value = data_values["max"].values[0]
            min_value = data_values["min"].values[0]

        start_time = time.time()
        for idx, slice_id in enumerate(slice_id_list):
            image = cv2.imread(
                os.path.join(data_dir, data_name, "images", f"{slice_id}.tif"),
                cv2.IMREAD_GRAYSCALE,
            )

            image = torch.from_numpy(image).float()
            image = clip_underover_value(image)
            image = image.numpy()
            if max_min_df is not None:
                image = min_max_normalize_img(image, min_value, max_value)

            pred, _ = model.overlap_predict(image)
            pred = remove_small_objects(pred, object_min_size, threshold)
            rle = rle_encode(pred)
            data_id_list.append(f"{data_name}_{slice_id}")
            rle_list.append(rle)
            if idx % 100 == 0:
                elapsed_time = (time.time() - start_time) / 60
                # print(f"\r{idx:04d}/{len(slice_id_list)}", end="")
                print(
                    f"\r{idx:04d}/{len(slice_id_list)}, {elapsed_time:.2f} [min]",
                    end="",
                )

    submit_df = pd.DataFrame({"id": data_id_list, "rle": rle_list})

    return submit_df
