import torch.nn as nn


class L1Loss(nn.Module):
    """Cross Entroy Loss"""

    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, prediction, target):
        target = target if len(target.shape) == 1 else target.squeeze(1)
        return self.criterion(prediction, target)
