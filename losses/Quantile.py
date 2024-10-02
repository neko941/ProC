import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self):
        super(QuantileLoss, self).__init__()

    def forward(self, predictions, targets,quantile=0.5):
        error = targets - predictions
        loss = torch.max((quantile - 1) * error, quantile * error)
        return loss.mean()