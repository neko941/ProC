import torch
import torch.nn as nn

class RMSLELoss(nn.Module):
    def __init__(self):
        super(RMSLELoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.sqrt(torch.mean((torch.log(targets+1)-torch.log(predictions+1))**2))