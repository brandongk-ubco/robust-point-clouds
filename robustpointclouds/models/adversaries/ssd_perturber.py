from torch import nn
from ..builder import ADVERSARIES


@ADVERSARIES.register_module()
class ssdPerturber(nn.Module):

    def __init__(self):
        super(ssdPerturber, self).__init__()

        self.model = nn.Sequential(nn.InstanceNorm2d(12), nn.Conv2d(12, 12, 1),
                                   nn.BatchNorm2d(12), nn.ReLU(),
                                   nn.Conv2d(12, 24, 1), nn.BatchNorm2d(24),
                                   nn.ReLU(), nn.Conv2d(24, 48, 1),
                                   nn.BatchNorm2d(48), nn.ReLU(),
                                   nn.Conv2d(48, 96, 1), nn.BatchNorm2d(96),
                                   nn.ReLU(), nn.Conv2d(96, 48, 1),
                                   nn.BatchNorm2d(48), nn.ReLU(),
                                   nn.Conv2d(48, 24, 1), nn.BatchNorm2d(24),
                                   nn.ReLU(), nn.Conv2d(24, 12, 1),
                                   nn.BatchNorm2d(12), nn.ReLU(),
                                   nn.Conv2d(12, 12, 1), nn.BatchNorm2d(12),
                                   nn.ReLU(), nn.Conv2d(12, 12, 1))

    def forward(self, points):
        x = points.clone()
        # x = x.transpose(0, 1)
        # import pdb
        # pdb.set_trace()
        x = x.unsqueeze(0)

        perturbation = self.model(x)
        perturbation = perturbation.squeeze()
        # perturbation = perturbation.transpose(0, 1)

        return perturbation
