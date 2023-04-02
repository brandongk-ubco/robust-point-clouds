import argh
from robustpointclouds.models.detectors.adversarial_voxel_net import AdversarialVoxelNet as _
from robustpointclouds.models.adversaries.voxel_perturber import VoxelPerturber as _
from robustpointclouds.models.detectors.adversarial_ssd3d_net import Adversarialssd3dnet as _
from robustpointclouds.models.adversaries.ssd_perturber import ssdPerturber as _
from robustpointclouds.models.detectors.adversarial_parta2 import AdversarialPartA2 as _
from robustpointclouds.commands import evaluate_loss, predict, visualize_loss

parser = argh.ArghParser()
parser.add_commands([evaluate_loss, predict, visualize_loss])

if __name__ == '__main__':
    parser.dispatch()
