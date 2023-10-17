import pytorch_lightning as pl
from mmengine.config import Config
from mmdet3d.registry import DATASETS
from mmengine.runner import Runner
import torch
from functools import partial
import copy


class mmdetection3dDataModule(pl.LightningDataModule):

    def __init__(self, config_file: str, batch_size: int = 2, num_workers=8):
        super().__init__()
        self.cfg = Config.fromfile(config_file)

    def train_dataloader(self):
        return Runner.build_dataloader(self.cfg.train_dataloader)

    def val_dataloader(self):
        return Runner.build_dataloader(self.cfg.val_dataloader)

    def test_dataloader(self):
        return Runner.build_dataloader(self.cfg.test_dataloader)

__all__ = ["mmdetection3dDataModule"]
