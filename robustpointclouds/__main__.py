import argh
from robustpointclouds.models.detectors.adversarial_voxel_net import AdversarialVoxelNet as _
from robustpointclouds.models.adversaries.voxel_perturber import VoxelPerturber as _
from robustpointclouds.commands import evaluate_loss

parser = argh.ArghParser()
parser.add_commands([evaluate_loss])

if __name__ == '__main__':
    parser.dispatch()
