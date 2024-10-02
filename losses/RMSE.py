import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets) ** 2))