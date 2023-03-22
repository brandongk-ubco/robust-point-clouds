from mmdet3d.models.detectors.ssd3dnet import SSD3DNet
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .. import builder
import torch


@DETECTORS.register_module()
class Adversarialssd3dnet(SSD3DNet):

    def __init__(self, adversary, **kwargs):
        super(Adversarialssd3dnet, self).__init__(**kwargs)
        self.adversary = builder.build_adversary(adversary)

