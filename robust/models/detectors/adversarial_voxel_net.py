from mmdet3d.models.detectors.voxelnet import VoxelNet
from mmdet.models import DETECTORS
from .. import builder


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
        return x
