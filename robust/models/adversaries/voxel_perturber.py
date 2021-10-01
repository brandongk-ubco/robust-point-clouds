from torch import nn
from ..builder import ADVERSARIES


@ADVERSARIES.register_module()
class VoxelPerturber(nn.Module):

    def __init__(self):
        super(VoxelPerturber, self).__init__()

        self.model = nn.Sequential(nn.InstanceNorm1d(4), nn.Conv1d(4, 4, 1),
                                   nn.Conv1d(4, 8, 1), nn.ReLU(),
                                   nn.Conv1d(8, 16, 1), nn.ReLU(),
                                   nn.Conv1d(16, 32, 1), nn.ReLU(),
                                   nn.Conv1d(32, 16, 1), nn.ReLU(),
                                   nn.Conv1d(16, 8, 1), nn.ReLU(),
                                   nn.Conv1d(8, 4, 1), nn.Conv1d(4, 4, 1),
                                   nn.Conv1d(4, 4, 1))

    def forward(self, points):
        x = points.clone()
        x = x.transpose(0, 1)
        x = x.unsqueeze(0)
        perturbation = self.model(x)
        perturbation = perturbation.squeeze()
        perturbation = perturbation.transpose(0, 1)

        return perturbation
