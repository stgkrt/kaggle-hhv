import os
from dataclasses import asdict

import lightning as L
from lightning import seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from src.conf import ExpConfig
from src.data.data_module import DataModule
from src.model.model_module import ModelModule


def run_train(config: ExpConfig) -> None:
    config = ExpConfig()
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    elif config.debug:
        os.makedirs(config.save_dir, exist_ok=True)
    else:
        raise RuntimeError(f"{config.save_dir} already exists")
    seed_everything(config.seed)

    datamodule = DataModule(config)
    model = ModelModule(config)
    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=config.save_dir,
        verbose=True,
        monitor=config.monitor,
        mode=config.monitor_mode,
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
    )
    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    # init experiment logger
    pl_logger = WandbLogger(
        project=config.competition_name,
        group=config.exp_category,
        name=config.exp_name,
        log_model=False,
        offline=True,
    )
    pl_logger.log_hyperparams(asdict(config))
    csv_logger = CSVLogger(save_dir=config.save_dir, name="log")
    csv_logger.log_hyperparams(asdict(config))

    trainer = L.Trainer(
        max_epochs=config.epochs,
        callbacks=[checkpoint_cb, lr_monitor, progress_bar, model_summary],
        logger=[pl_logger, csv_logger],
        sync_batchnorm=True,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
    )
    trainer.fit(model, datamodule=datamodule)

    return


if __name__ == "__main__":
    config = ExpConfig()
    run_train(config)
