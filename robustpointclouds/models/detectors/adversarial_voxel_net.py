from mmdet3d.models.detectors.voxelnet import VoxelNet
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .. import builder
import torch


@DETECTORS.register_module()
class AdversarialVoxelNet(VoxelNet):

    def __init__(self, adversary, **kwargs):
        super(AdversarialVoxelNet, self).__init__(**kwargs)
        self.adversary = builder.build_adversary(adversary)

    def extract_feat(self, points, img_metas=None):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        perturbation = self.adversary(voxel_features)
        voxel_features = voxel_features + perturbation
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x, perturbation

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x, _ = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats, _ = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(*outs,
                                                  img_meta,
                                                  rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x, perturbation = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(*loss_inputs,
                                     gt_bboxes_ignore=gt_bboxes_ignore)

        losses["perturbation_norm"] = torch.mean(
            torch.linalg.vector_norm(perturbation, dim=1, ord=2))

        losses["perturbation_bias"] = torch.linalg.vector_norm(torch.mean(
            perturbation, dim=0),
                                                               ord=2)
        losses["perturbation_imbalance"] = torch.std(
            torch.mean(perturbation, dim=0))

        return losses, perturbation
