import pytorch_lightning as pl
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmcv.parallel import collate
import torch
from functools import partial
import copy


class mmdetection3dDataModule(pl.LightningDataModule):

    def __init__(self, config_file: str, batch_size: int = 2, num_workers=8):
        super().__init__()
        self.cfg = Config.fromfile(config_file)
        self.batch_size = batch_size
        self.num_workers = num_workers

        val_config = copy.deepcopy(self.cfg.data.val)

        if 'dataset' in self.cfg.data.train:
            val_config.pipeline = self.cfg.data.train.dataset.pipeline
        else:
            val_config.pipeline = self.cfg.data.train.pipeline

        val_config.test_mode = False

        self.train_dataset = build_dataset(self.cfg.data.train)
        self.val_dataset = build_dataset(val_config)
        self.test_dataset = build_dataset(self.cfg.data.test)

    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=partial(collate, samples_per_gpu=self.batch_size))
        return data_loader

    def val_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=partial(collate, samples_per_gpu=self.batch_size))
        return data_loader

    def test_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=partial(collate, samples_per_gpu=1))
        return data_loader


__all__ = ["mmdetection3dDataModule"]
