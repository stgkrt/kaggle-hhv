import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExpConfig:
    # common
    debug: bool = False
    phase: str = "train"
    # experiment
    # exp_name: str = "exp002_Gradloss"
    exp_name: str = "exp009_512_train13_percentail"
    # exp_name: str = "exp003_Gradloss05"
    exp_category: str = "baseline"
    seed: int = 42
    # model
    model_name: str = "SegModel"
    # encoder_name: str = "tf_efficientnet_b0"
    encoder_name: str = "seresnext50_32x4d"
    pretrained: bool = True
    in_channels: int = 1
    out_channels: int = 1
    use_batchnorm: bool = True
    dropout: float = 0.3
    encoder_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256, 512, 512]
    )
    decoder_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 64])
    # dirs
    input_dir: str = "/kaggle/input"
    competition_name: str = "blood-vessel-segmentation"
    input_data_dir: str = os.path.join(input_dir, competition_name)
    processed_data_dir: str = os.path.join("/kaggle", "working", "_processed")
    output_dir: str = "/kaggle/working"
    save_dir: str = os.path.join(output_dir, exp_name)
    save_weight_dir = os.path.join(output_dir, "weights", exp_name)
    # data
    img_height: int = 512
    img_width: int = 512
    # img_height: int = 1024
    # img_width: int = 1024
    ## preparedata
    stride_height: int = img_height
    stride_width: int = img_width
    patch_height: int = int(stride_height * 1.5)
    patch_width: int = int(stride_width * 1.5)
    ## loader
    slice_num: int = in_channels
    batch_size: int = 32
    num_workers: int = 2
    train_df: str = os.path.join(
        output_dir, f"train_{stride_height}_{stride_width}.csv"
    )
    valid_df: str = os.path.join(
        output_dir, f"valid_{stride_height}_{stride_width}.csv"
    )
    label_df = os.path.join(input_data_dir, "train_rles.csv")
    train_data_name: List[str] = field(
        default_factory=lambda: [
            "kidney_1_dense",
            # "kidney_1_voi",
            # "kidney_2",
            "kidney_3_sparse",
        ]
    )
    valid_data_name: List[str] = field(
        default_factory=lambda: [
            # "kidney_1_dense",
            "kidney_2"
        ]
    )
    minmax_df_path: str = os.path.join(output_dir, "centerslice_maxmean.csv")

    # train
    epochs: int = 10
    T_max: int = epochs
    lr: float = 1e-4
    eta_min: float = 1e-6
    loss_type: str = "dice_grad"
    # logger
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    check_val_every_n_epoch: int = 1
    # predict
    overlap_rate: float = 0.2
    threshold: float = 0.5
    object_min_size: int = 3
    if encoder_name == "seresnext50_32x4d":
        batch_size = 2
    if debug:
        exp_name = "debug"
        exp_category = "debug"
        save_dir = os.path.join(output_dir, exp_name)
        save_weight_dir = os.path.join(output_dir, "weights", exp_name)
        train_df = os.path.join(
            output_dir, f"train_{stride_height}_{stride_width}_debug.csv"
        )
        valid_df = os.path.join(
            output_dir, f"valid_{stride_height}_{stride_width}_debug.csv"
        )
        epochs = 2
        train_data_name = field(default_factory=lambda: ["kidney_1_dense"])
        valid_data_name = field(default_factory=lambda: ["kidney_2"])


if __name__ == "__main__":
    config = ExpConfig()
    print(config)
    print("---")
    from dataclasses import asdict

    config_dict = asdict(config)
    print(config_dict)
