import torch
import torch.nn as nn

class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean((torch.log(targets+1)-torch.log(predictions+1))**2)