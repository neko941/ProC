import torch
import torch.nn as nn

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.abs(targets-predictions)/targets)*100