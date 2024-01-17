import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExpConfig:
    # common
    debug: bool = False
    phase: str = "train"
    # experiment
    exp_name: str = "exp001_validscore"
    # exp_name: str = "exp002_seresnext50_32x4d"
    exp_category: str = "baseline"
    seed: int = 42
    # dirs
    input_dir: str = "/kaggle/input"
    competition_name: str = "blood-vessel-segmentation"
    input_data_dir: str = os.path.join(input_dir, competition_name)
    processed_data_dir: str = os.path.join("/kaggle", "working", "_processed")
    output_dir: str = "/kaggle/working"
    save_dir: str = os.path.join(output_dir, exp_name)
    # data
    img_height: int = 512
    img_width: int = 512
    ## preparedata
    stride_height: int = img_height
    stride_width: int = img_width
    patch_height: int = int(stride_height * 1.5)
    patch_width: int = int(stride_width * 1.5)
    ## loader
    slice_num: int = 1
    batch_size: int = 32
    # batch_size: int = 8
    num_workers: int = 2
    train_df: str = os.path.join(
        output_dir, f"train_{stride_height}_{stride_width}.csv"
    )
    valid_df: str = os.path.join(
        output_dir, f"valid_{stride_height}_{stride_width}.csv"
    )
    label_df = os.path.join(input_data_dir, "train_rles.csv")
    train_data_name: List[str] = field(default_factory=lambda: ["kidney_1_dense"])
    valid_data_name: List[str] = field(default_factory=lambda: ["kidney_2"])
    minmax_df_path: str = os.path.join(output_dir, "centerslice_maxmean.csv")
    # model
    model_name: str = "SegModel"
    encoder_name: str = "tf_efficientnet_b0"
    # encoder_name: str = "seresnext50_32x4d"
    pretrained: bool = True
    in_channels: int = slice_num
    out_channels: int = slice_num
    use_batchnorm: bool = True
    dropout: float = 0.2
    encoder_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 512]
    )
    decoder_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 64])
    # train
    epochs: int = 10
    T_max: int = epochs
    lr: float = 1e-4
    eta_min: float = 1e-6
    # logger
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    check_val_every_n_epoch: int = 1
    # predict
    overlap_rate: float = 0.2

    if debug:
        exp_name = "debug"
        exp_category = "debug"
        train_df = os.path.join(
            output_dir, f"train_{stride_height}_{stride_width}_debug.csv"
        )
        valid_df = os.path.join(
            output_dir, f"valid_{stride_height}_{stride_width}_debug.csv"
        )
        epochs = 2
        train_data_name = field(default_factory=lambda: ["kidney_1_voi"])
        valid_data_name = field(default_factory=lambda: ["kidney_3_sparse"])


if __name__ == "__main__":
    config = ExpConfig()
    print(config)
    print("---")
    from dataclasses import asdict

    config_dict = asdict(config)
    print(config_dict)
