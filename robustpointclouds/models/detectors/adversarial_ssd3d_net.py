from mmdet3d.models.detectors.ssd3dnet import SSD3DNet
# from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .. import builder
import torch


@DETECTORS.register_module()
class Adversarialssd3dnet(SSD3DNet):

    def __init__(self, adversary, **kwargs):
        super(Adversarialssd3dnet, self).__init__(**kwargs)
        self.adversary = builder.build_adversary(adversary)

    def extract_feat(self, points, img_metas=None):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        perturbation = self.adversary(points)
        points = points + perturbation
        x = self.backbone(points)
        if self.with_neck:
            x = self.neck(x)
        return x, perturbation

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """
        points_cat = torch.stack(points)

        x, perturbation = self.extract_feat(points_cat)
        bbox_preds = self.bbox_head(x, self.train_cfg.sample_mod)
        loss_inputs = (points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                       pts_instance_mask, img_metas)
        losses = self.bbox_head.loss(bbox_preds,
                                     *loss_inputs,
                                     gt_bboxes_ignore=gt_bboxes_ignore)
        losses["perturbation_norm"] = torch.mean(
            torch.linalg.vector_norm(perturbation, dim=1, ord=2))

        losses["perturbation_bias"] = torch.linalg.vector_norm(torch.mean(
            perturbation, dim=0),
                                                               ord=2)
        losses["perturbation_imbalance"] = torch.std(
            torch.mean(perturbation, dim=0))

        return losses, perturbation
