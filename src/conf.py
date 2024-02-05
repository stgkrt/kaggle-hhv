import dataclasses
import inspect
import pathlib
from dataclasses import dataclass
from typing import List

import yaml  # type: ignore


@dataclasses.dataclass
class YamlConfig:
    def save(self, config_path: pathlib.Path):
        """Export config as YAML file"""
        assert (
            config_path.parent.exists()
        ), f"directory {config_path.parent} does not exist"

        def convert_dict(data):
            for key, val in data.items():
                if isinstance(val, pathlib.Path):
                    data[key] = str(val)
                if isinstance(val, dict):
                    data[key] = convert_dict(val)
            return data

        with open(config_path, "w") as f:
            yaml.dump(convert_dict(dataclasses.asdict(self)), f)

    @classmethod
    def load(cls, config_path: str):
        def convert_from_dict(parent_cls, data):
            for key, val in data.items():
                child_class = parent_cls.__dataclass_fields__[key].type
                if inspect.isclass(child_class) and issubclass(child_class, YamlConfig):
                    data[key] = child_class(**convert_from_dict(child_class, val))
            return data

        with open(config_path) as f:
            config_data = yaml.full_load(f)
            # recursively convert config item to YamlConfig
            config_data = convert_from_dict(cls, config_data)
            return cls(**config_data)


@dataclass
class ExpConfig(YamlConfig):
    # common
    debug: bool
    phase: str
    # experiment
    exp_name: str
    exp_category: str
    seed: int
    # model
    model_name: str
    encoder_name: str
    pretrained: bool
    in_channels: int
    out_channels: int
    use_batchnorm: bool
    dropout: float
    encoder_channels: List[int]
    decoder_channels: List[int]
    # dirs
    input_dir: str
    competition_name: str
    input_data_dir: str
    processed_data_dir: str
    output_dir: str
    save_dir: str
    save_weight_dir: str
    # data
    img_height: int
    img_width: int
    ## preparedata
    stride_height: int
    stride_width: int
    patch_height: int
    patch_width: int
    ## loader
    slice_num: int
    batch_size: int
    num_workers: int
    train_df: str
    valid_df: str
    label_df: str
    train_data_name: List[str]
    valid_data_name: List[str]
    minmax_df_path: str | None

    # train
    epochs: int
    T_max: int
    lr: float
    eta_min: float
    loss_type: str
    # logger
    monitor: str
    monitor_mode: str
    check_val_every_n_epoch: int
    # predict
    overlap_rate: float
    threshold: float
    object_min_size: int


if __name__ == "__main__":
    config = ExpConfig.load("/kaggle/config/config.yaml")
    print(config)
