import os
from dataclasses import dataclass


@dataclass
class ExpConfig:
    # common
    debug = True
    phase = "train"
    # experiment
    if debug:
        exp_name = "debug"
        exp_category = "debug"
    else:
        exp_name = "exp000"
        exp_category = "baseline"
    seed = 42
    # dirs
    input_dir = "/kaggle/input"
    competition_name = "blood-vessel-segmentation"
    input_data_dir = os.path.join(input_dir, competition_name)
    processed_data_dir = os.path.join("/kaggle", "working", "_processed")
    output_dir = "/kaggle/working"
    save_dir = os.path.join(output_dir, exp_name)
    # data
    img_height = 256
    img_width = 256
    slice_num = 3
    batch_size = 32
    shuffle = True
    num_workers = 2
    # model
    model_name = "SegModel"
    encoder_name = "resnet18"
    pretrained = True
    in_channels = slice_num
    out_channels = slice_num
    use_batchnorm = True
    dropout = 0.2
    encoder_channels = [64, 64, 128, 256, 512]
    decoder_channels = [512, 256, 128, 64, 64]
    # train
    epochs = 10
    if debug:
        epochs = 5
    lr = 1e-3
    T_max = epochs
    eta_min = 1e-5
    # logger
    monitor = "val_loss"
    monitor_mode = "min"
    check_val_every_n_epoch = 1
