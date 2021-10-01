import pytorch_lightning as pl
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmcv.parallel import collate
import torch
from functools import partial


class mmdetection3dDataModule(pl.LightningDataModule):

    def __init__(self, config_file: str, batch_size: int = 8, num_workers=8):
        super().__init__()
        self.cfg = Config.fromfile(config_file)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = build_dataset(self.cfg.data.train)
        self.val_dataset = build_dataset(self.cfg.data.train)
        self.test_dataset = build_dataset(self.cfg.data.train)

    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=partial(collate, samples_per_gpu=self.batch_size))
        return data_loader

    def test_dataloader(self):
        data_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=1,
                                                  num_workers=self.num_workers,
                                                  shuffle=False,
                                                  drop_last=True,
                                                  pin_memory=True,
                                                  collate_fn=partial(
                                                      collate,
                                                      samples_per_gpu=1))
        return data_loader


__all__ = [mmdetection3dDataModule]
