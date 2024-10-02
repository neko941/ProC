import torch
import torch.nn as nn

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.sum(torch.log(torch.cosh(predictions-targets)))