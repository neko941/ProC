import torch
import torch.nn as nn

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.abs(predictions - targets))