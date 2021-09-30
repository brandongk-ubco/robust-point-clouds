import pytorch_lightning as pl
from mmcv import Config
from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmseg import __version__ as mmseg_version
from mmdet3d.models import build_model
from mmcv.runner import build_optimizer
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset


class mmdetection3dLightningModule(pl.LightningModule):

    def __init__(self, config_file: str, checkpoint_file: str):
        super().__init__()
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

        self.model = self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.checkpoint = load_checkpoint(self.model, checkpoint_file)

        if 'CLASSES' in self.checkpoint['meta']:
            self.model.CLASSES = self.checkpoint['meta']['CLASSES']
        else:
            self.model.CLASSES = self.cfg.class_names
        if 'PALETTE' in self.checkpoint['meta']:  # 3D Segmentor
            self.model.PALETTE = self.checkpoint['meta']['PALETTE']

    def forward(self, sample):

        data = {
            "points": sample["points"].data[0],
            "img_metas": sample["img_metas"].data[0],
            "gt_labels_3d": sample["gt_labels_3d"].data[0],
            "gt_bboxes_3d": sample["gt_bboxes_3d"].data[0]
        }
        for i in data["img_metas"]:
            i["pcd_rotation"] = i["pcd_rotation"].to(self.device)

        for idx_i, i in enumerate(data["points"]):
            data["points"][idx_i] = i.to(self.device)

        for idx_i, i in enumerate(data["gt_labels_3d"]):
            data["gt_labels_3d"][idx_i] = i.long().to(self.device)

        for idx_i, i in enumerate(data["gt_bboxes_3d"]):
            data["gt_bboxes_3d"][idx_i] = i.to(self.device)

        for idx_i in range(len(data["gt_labels_3d"])):
            valid_gt = data["gt_labels_3d"][idx_i] != -1
            data["gt_bboxes_3d"][idx_i] = data["gt_bboxes_3d"][idx_i][valid_gt]
            data["gt_labels_3d"][idx_i] = data["gt_labels_3d"][idx_i][valid_gt]

        result = self.model(return_loss=True, **data)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)

    def configure_optimizers(self):
        return build_optimizer(self.model, self.cfg.optimizer)


__all__ = [mmdetection3dLightningModule]