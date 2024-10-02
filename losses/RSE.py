import torch
import torch.nn as nn

class RSELoss(nn.Module):
    def __init__(self):
        super(RSELoss, self).__init__()

    def forward(self, predictions, targets):
        numerator = torch.sum((targets-predictions)**2)
        denominator = torch.sum((targets-torch.mean(targets))**2)
        return torch.divide(numerator,denominator)