import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExpConfig:
    # common
    debug: bool = True
    phase: str = "train"
    # experiment
    if debug:
        exp_name: str = "debug"
        exp_category: str = "debug"
    else:
        exp_name: str = "exp000"  # type: ignore
        exp_category: str = "baseline"  # type: ignore
    seed: int = 42
    # dirs
    input_dir: str = "/kaggle/input"
    competition_name: str = "blood-vessel-segmentation"
    input_data_dir: str = os.path.join(input_dir, competition_name)
    processed_data_dir: str = os.path.join("/kaggle", "working", "_processed")
    output_dir: str = "/kaggle/working"
    save_dir: str = os.path.join(output_dir, exp_name)
    # data
    img_height: int = 256
    img_width: int = 256
    slice_num: int = 3
    batch_size: int = 32
    num_workers: int = 2
    train_data_name: List[str] = field(default_factory=lambda: ["kidney_1_voi"])
    valid_data_name: List[str] = field(default_factory=lambda: ["kidney_3_sparse"])
    # model
    model_name: str = "SegModel"
    encoder_name: str = "resnet18"
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
    if debug:
        epochs = 2
    T_max: int = epochs
    lr: float = 1e-3
    eta_min: float = 1e-8
    # logger
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    check_val_every_n_epoch: int = 1


if __name__ == "__main__":
    config = ExpConfig()
    print(config)
    print("---")
    from dataclasses import asdict

    config_dict = asdict(config)
    print(config_dict)
