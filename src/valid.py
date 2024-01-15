import os
import sys
from glob import glob
from typing import List

import cv2
import numpy as np
import pandas as pd
from lightning import LightningModule

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from score.rle_convert import rle_encode


def make_submit_df(
    model: LightningModule,
    data_name_list: List[str],
    data_dir: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    data_id_list = []
    rle_list = []
    for data_name in data_name_list:
        print("predicting...", data_name)

        slice_id_list = sorted(
            [
                os.path.basename(path).split(".")[0]
                for path in glob(os.path.join(data_dir, f"{data_name}/images/*.tif"))
            ]
        )
        for slice_id in slice_id_list:
            print("all slice num:", len(slice_id_list))
            print("\r", slice_id, end="")
            image = cv2.imread(
                os.path.join(data_dir, f"{data_name}/images/{slice_id}.tif"),
                cv2.IMREAD_GRAYSCALE,
            )
            pred, pred_counts = model.overlap_predict(image)
            rle = rle_encode((pred > threshold).astype(np.uint8))
            data_id_list.append(f"{data_name}_{slice_id}")
            rle_list.append(rle)
    submit_df = pd.DataFrame({"id": data_id_list, "rle": rle_list})

    return submit_df
