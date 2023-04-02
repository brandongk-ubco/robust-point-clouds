import pytorch_lightning as pl
from mmcv import Config
from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmseg import __version__ as mmseg_version
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
import torch
import tensorflow as tf
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from mmdet3d.apis.inference import inference_detector


class mmdetection3dLightningModule(pl.LightningModule):

    def __init__(self,
                 config_file: str,
                 checkpoint_file: str,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 5e-3,
                 perturbation_norm_regularizer: float = 3.,
                 perturbation_bias_regularizer: float = 10.,
                 perturbation_imbalance_regularizer: float = 10.):
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

        self.model.cfg = self.cfg

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
            ModelCheckpoint(save_last=True, filename='{epoch}-{val_loss:.6f}'),
        ]

        try:
            callbacks.append(DeviceStatsMonitor())
        except MisconfigurationException:
            pass
        return callbacks

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)

        return {"optimizer": optimizer}

    def forward(self, sample, return_loss=False):

        data = {
            "points": sample["points"].data[0],
            "img_metas": sample["img_metas"].data[0],
            "gt_labels_3d": sample["gt_labels_3d"].data[0],
            "gt_bboxes_3d": sample["gt_bboxes_3d"].data[0]
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
        result = self(batch, return_loss=True)
        if type(result) is tuple:
            losses, perturbation = result
        else:
            losses = result
            perturbation = None

        perturbation_norm = losses.pop("perturbation_norm")
        perturbation_bias = losses.pop("perturbation_bias")
        perturbation_imbalance = losses.pop("perturbation_imbalance")

        loss_values = []
        for _, loss in losses.items():
            loss_values += loss
            # loss_values.append(loss)
            if len(loss_values) == 3:
                break
        model_loss = torch.sum(torch.stack(loss_values))

        self.log_dict(
            {
                "model_loss": model_loss,
                "ptb_norm": perturbation_norm,
                "ptb_bias": perturbation_bias,
                "ptb_imbalance": perturbation_imbalance
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=self.trainer.datamodule.batch_size)
        loss = -model_loss
        loss += (perturbation_norm * self.hparams.perturbation_norm_regularizer)
        loss += (perturbation_bias * self.hparams.perturbation_bias_regularizer)
        loss += (perturbation_imbalance *
                 self.hparams.perturbation_imbalance_regularizer)
        return loss

    def validation_step(self, batch, batch_idx):
        result = self(batch, return_loss=True)
        if type(result) is tuple:
            losses, perturbation = result
        else:
            losses = result
            perturbation = None

        perturbation_norm = losses.pop("perturbation_norm")
        perturbation_bias = losses.pop("perturbation_bias")
        perturbation_imbalance = losses.pop("perturbation_imbalance")
        loss_values = []
        for loss_type, loss in losses.items():
            # if tf.rank(loss) != 0:
            # import pdb
            # pdb.set_trace()
            loss_values += loss
            # loss_values.append(loss)
            if len(loss_values) == 3:
                break
        model_loss = torch.sum(torch.stack(loss_values))

        val_loss = -model_loss
        val_loss += (perturbation_norm *
                     self.hparams.perturbation_norm_regularizer)
        val_loss += (perturbation_bias *
                     self.hparams.perturbation_bias_regularizer)
        val_loss += (perturbation_imbalance *
                     self.hparams.perturbation_imbalance_regularizer)

        self.log_dict({
            "val_model_loss": model_loss,
            "val_loss": val_loss
        },
                      prog_bar=True,
                      on_step=False,
                      on_epoch=True,
                      batch_size=self.trainer.datamodule.batch_size)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            result = self(batch, return_loss=True)
            if type(result) is tuple:
                losses, perturbation = result
            else:
                losses = result
                perturbation = None
        return losses, perturbation

    def predict_file(self, filename):
        return inference_detector(self.model, filename)


__all__ = ["mmdetection3dLightningModule"]
