import torch
import torch.nn as nn

class RAELoss(nn.Module):
    def __init__(self):
        super(RAELoss, self).__init__()

    def forward(self, predictions, targets):
        numerator = torch.sum(torch.abs(targets-predictions))
        denominator = torch.sum(torch.abs(targets-torch.mean(targets)))
        return torch.divide(numerator,denominator)