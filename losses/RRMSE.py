import torch
import torch.nn as nn

class RRMSELoss(nn.Module):
    def __init__(self):
        super(RRMSELoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets) ** 2)/torch.sum(predictions**2))