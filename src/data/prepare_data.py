import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.conf import ExpConfig


def get_data_name_list(config: ExpConfig, phase: str = "train") -> list[str]:
    input_data_dir = config.input_data_dir
    if phase == "train":
        load_phase = "train"
    elif phase == "valid":
        load_phase = "train"
    else:
        load_phase = "test"
    data_name_list = os.listdir(os.path.join(input_data_dir, load_phase))
    data_name_list = [data_name.split(".")[0] for data_name in data_name_list]
    return sorted(data_name_list)


def get_file_name_list(
    config: ExpConfig,
    data_name: str,
    data_type: str = "images",
    is_processed: bool = False,
) -> list[str]:
    if is_processed:
        if config.phase == "train":
            phase = "train"
        elif config.phase == "valid":
            phase = "train"
        else:
            phase = "test"
        data_dir = (
            f"{config.processed_data_dir}_{config.stride_height}_{config.stride_width}"
        )
        print(data_dir)
        file_name_list = glob(
            os.path.join(data_dir, phase, data_name, data_type, "*.npy")
        )
    else:
        if config.phase == "train" or config.phase == "valid":
            phase = "train"
        else:
            phase = "test"
        data_dir = config.input_data_dir
        file_name_list = glob(
            os.path.join(data_dir, phase, data_name, data_type, "*.tif")
        )
    file_name_list = [file_name.split("/")[-1] for file_name in file_name_list]
    return sorted(file_name_list)


def set_save_and_load_phase(phase: str = "train") -> tuple[str, str]:
    if phase == "train":
        load_phase = "train"
        save_phase = "train"
    elif phase == "valid":
        load_phase = "train"
        save_phase = "valid"
    else:
        load_phase = "test"
        save_phase = "test"
    return load_phase, save_phase


def save_split_patch_image(config: ExpConfig, phase: str = "train"):
    """save splited patch image and label from original image and label

    Args:
        config (ExpConifg): _description_
    """
    # dataloaderでrandomで切り出せるようにsizeは大きめにとる
    patch_height = config.patch_height
    patch_width = config.patch_width
    # stride分ずらしながら切り出す
    stride_height = config.stride_height
    stride_width = config.stride_width
    load_phase, save_phase = set_save_and_load_phase(phase)
    data_dir = config.input_data_dir
    processed_data_dir = f"{config.processed_data_dir}_{stride_height}_{stride_width}"
    data_name_list = get_data_name_list(config, phase)
    print(data_name_list)
    for data_name in data_name_list[1:]:
        print("processing... ->", "data_name:", data_name, "phase:", load_phase)
        file_name_list = get_file_name_list(config, data_name, "images")
        # set dir
        input_image_dir = os.path.join(data_dir, load_phase, data_name, "images")
        input_label_dir = os.path.join(data_dir, load_phase, data_name, "labels")
        print("input_image_dir:", input_image_dir)
        print("input_label_dir:", input_label_dir)
        # set save dir
        processed_image_dir = os.path.join(
            processed_data_dir, save_phase, data_name, "images"
        )
        os.makedirs(processed_image_dir, exist_ok=True)
        processed_label_dir = os.path.join(
            processed_data_dir, save_phase, data_name, "labels"
        )
        os.makedirs(processed_label_dir, exist_ok=True)
        print("processed_image_dir:", processed_image_dir)
        print("processed_label_dir:", processed_label_dir)
        for file_name in tqdm(file_name_list):
            image = cv2.imread(
                os.path.join(input_image_dir, file_name),
                cv2.IMREAD_GRAYSCALE,
            )
            image = image.astype(np.float32) / 255
            label = cv2.imread(
                os.path.join(input_label_dir, file_name),
                cv2.IMREAD_GRAYSCALE,
            )
            label = (label > 0).astype(np.float32) / 255
            # patch sizeで分割
            h, w = image.shape
            h_split = h // stride_height
            w_split = w // stride_width
            # 中央を取り出せるように全体を少しstrideさせる
            h_stride_offset = (h - stride_height * h_split) // 2
            w_stride_offset = (w - stride_width * w_split) // 2
            for i in range(h_split):
                for j in range(w_split):
                    # patchの範囲を計算
                    this_patch_height_min = h_stride_offset + stride_height * i
                    this_patch_width_min = w_stride_offset + stride_width * j
                    # maxは画像サイズを超えないようにする
                    this_patch_height_max = this_patch_height_min + patch_height
                    this_patch_width_max = this_patch_width_min + patch_width
                    this_patch_height_max = min(this_patch_height_max, h)
                    this_patch_width_max = min(this_patch_width_max, w)

                    # patchを切り出す
                    image_patch = image[
                        this_patch_height_min:this_patch_height_max,
                        this_patch_width_min:this_patch_width_max,
                    ]

                    label_patch = label[
                        this_patch_height_min:this_patch_height_max,
                        this_patch_width_min:this_patch_width_max,
                    ]
                    patch_filename = file_name.split(".")[0] + f"_{i}_{j}"
                    np.save(
                        os.path.join(processed_image_dir, patch_filename),
                        image_patch,
                    )
                    np.save(
                        os.path.join(processed_label_dir, patch_filename),
                        label_patch,
                    )

    return


def make_dataset_df(config: ExpConfig) -> pd.DataFrame:
    data_name_list = get_data_name_list(config)
    dataset_df = pd.DataFrame()
    slice_num = config.slice_num
    for data_name in data_name_list[1:]:
        print("data name:", data_name)
        file_name_list = get_file_name_list(config, data_name, "images", True)
        original_file_name_list = get_file_name_list(config, data_name, "images", False)
        all_slice_num = len(original_file_name_list)
        print("all slice num:", all_slice_num)
        for file_name in tqdm(file_name_list):
            file_name = file_name.split(".")[0]
            file_slice_num = int(file_name.split("_")[0])
            patch_id = "_".join(file_name.split("_")[1:])
            if file_slice_num + slice_num < all_slice_num:
                data_dict = {}
                data_dict["data_name"] = data_name
                data_dict["phase"] = config.phase
                for i in range(slice_num):
                    this_file_slice_num = int(file_slice_num) + i
                    data_dict[
                        f"file_name_{i}"
                    ] = f"{this_file_slice_num:04d}_{patch_id}.npy"
                tmp_df = pd.DataFrame(data_dict, index=[0])
            else:
                break
            dataset_df = pd.concat([dataset_df, tmp_df], axis=0)
    return dataset_df


def run_prepare_data(config: ExpConfig) -> None:
    if config.phase == "train":
        print("processing train data")
        phase = "train"
        # save_split_patch_image(config, phase)
        df = make_dataset_df(config)
        df.to_csv(
            os.path.join(
                "/kaggle",
                "working",
                f"{phase}_{config.stride_height}_{config.stride_width}.csv",
            ),
            index=False,
        )
        make_debug_df(config, phase)
        print("processing valid data")
        phase = "valid"
        # save_split_patch_image(config, phase)
        df = make_dataset_df(config)
        df.to_csv(
            os.path.join(
                "/kaggle",
                "working",
                f"{phase}_{config.stride_height}_{config.stride_height}.csv",
            ),
            index=False,
        )
        make_debug_df(config, phase)
    else:
        print("processing test data")
        phase = "test"
        save_split_patch_image(config)
        df = make_dataset_df(config)
        df.to_csv(os.path.join("/kaggle", "working", f"{phase}.csv"), index=False)
    return


def make_debug_df(config: ExpConfig, phase="train"):
    df = pd.read_csv(
        os.path.join(
            "/kaggle",
            "working",
            f"{phase}_{config.stride_height}_{config.stride_width}.csv",
        )
    )
    print(df.head())

    print("data name unique", df["data_name"].unique())
    if phase == "train":
        use_data_name = "kidney_1_voi"
    else:
        use_data_name = "kidney_3_sparse"
    print("use data name:", use_data_name)
    df = df[df["data_name"] == use_data_name]
    df = df.sample(1000).reset_index(drop=True)
    df.to_csv(
        os.path.join(
            "/kaggle",
            "working",
            f"{phase}_{config.stride_height}_{config.stride_width}_debug.csv",
        ),
        index=False,
    )
    return


if __name__ == "__main__":
    config = ExpConfig()
    run_prepare_data(config)
