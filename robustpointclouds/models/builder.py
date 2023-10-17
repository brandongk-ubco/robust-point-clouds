from mmdet3d.registry import MODELS as MMCV_MODELS
from mmengine.registry import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

ADVERSARIES = MODELS


def build_adversary(cfg):
    """Build adversary."""
    return ADVERSARIES.build(cfg)
