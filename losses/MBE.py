import torch
import torch.nn as nn

class MBELoss(nn.Module):
    def __init__(self):
        super(MBELoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(predictions - targets)