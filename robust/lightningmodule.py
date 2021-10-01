import pytorch_lightning as pl
from mmcv import Config
from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmseg import __version__ as mmseg_version
from mmdet3d.models import build_model
import copy
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint, GPUStatsMonitor
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class mmdetection3dLightningModule(pl.LightningModule):

    def __init__(self,
                 config_file: str,
                 checkpoint_file: str,
                 patience: int = 3,
                 learning_rate: float = 5e-3,
                 min_learning_rate: float = 5e-4,
                 weight_decay: float = 0.0):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = Config.fromfile(config_file)

        self.dataset = build_dataset(self.cfg.data.train)

        self.cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=self.cfg.pretty_text,
            CLASSES=self.dataset.CLASSES,
            PALETTE=self.dataset.PALETTE  # for segmentors
            if hasattr(self.dataset, 'PALETTE') else None)

        self.model = build_model(self.cfg.model,
                                 train_cfg=self.cfg.get('train_cfg'),
                                 test_cfg=self.cfg.get('test_cfg'))

        for child in [
                c[1] for c in self.model.named_children() if c[0] != "adversary"
        ]:
            for param in child.parameters():
                param.requires_grad = False

        checkpoint = load_checkpoint(self.model, checkpoint_file)

        if 'CLASSES' in checkpoint['meta']:
            self.model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            self.model.CLASSES = self.cfg.class_names
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            self.model.PALETTE = checkpoint['meta']['PALETTE']

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            EarlyStopping(patience=2 * self.hparams.patience,
                          monitor='val_loss',
                          verbose=True,
                          mode='min'),
            ModelCheckpoint(monitor='val_loss',
                            save_top_k=1,
                            mode="min",
                            filename='{epoch}-{val_loss:.6f}'),
        ]

        try:
            callbacks.append(GPUStatsMonitor())
        except MisconfigurationException:
            pass
        return callbacks

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.hparams.patience,
            min_lr=self.hparams.min_learning_rate,
            verbose=True,
            mode="min")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def forward(self, sample, return_loss=False):

        data = copy.deepcopy(sample)

        data = {
            "points": data["points"].data[0],
            "img_metas": data["img_metas"].data[0],
            "gt_labels_3d": data["gt_labels_3d"].data[0],
            "gt_bboxes_3d": data["gt_bboxes_3d"].data[0]
        }

        for i in data["img_metas"]:
            i["pcd_rotation"] = i["pcd_rotation"].clone().to(self.device)

        for idx_i, i in enumerate(data["points"]):
            data["points"][idx_i] = i.clone().to(self.device)

        for idx_i, i in enumerate(data["gt_labels_3d"]):
            data["gt_labels_3d"][idx_i] = i.long().clone().to(self.device)

        for idx_i, i in enumerate(data["gt_bboxes_3d"]):
            data["gt_bboxes_3d"][idx_i] = i.clone().to(self.device)

        for idx_i in range(len(data["gt_labels_3d"])):
            valid_gt = data["gt_labels_3d"][idx_i] != -1
            data["gt_bboxes_3d"][idx_i] = data["gt_bboxes_3d"][idx_i][valid_gt]
            data["gt_labels_3d"][idx_i] = data["gt_labels_3d"][idx_i][valid_gt]

        if not return_loss:
            del data["gt_labels_3d"]
            del data["gt_bboxes_3d"]

        results = self.model(return_loss=return_loss, **data)

        return results

    def training_step(self, batch, batch_idx):
        results = self(batch, return_loss=True)
        losses = []
        for loss_type, loss in results.items():
            losses += loss
        model_loss = torch.sum(torch.stack(losses))

        return model_loss

    def validation_step(self, batch, batch_idx):
        results = self(batch, return_loss=True)

        losses = []
        for loss_type, loss in results.items():
            losses += loss
        model_loss = torch.sum(torch.stack(losses))

        self.log('val_loss',
                 model_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)


__all__ = [mmdetection3dLightningModule]
