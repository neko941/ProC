import torch
import torch.nn as nn

class NRMSELoss(nn.Module):
    def __init__(self):
        super(NRMSELoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets) ** 2))/torch.mean(targets)