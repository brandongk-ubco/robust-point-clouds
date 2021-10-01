from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

ADVERSARIES = MODELS


def build_adversary(cfg):
    """Build adversary."""
    return ADVERSARIES.build(cfg)
