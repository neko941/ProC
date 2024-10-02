import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self):
        super(HuberLoss, self).__init__()

    def forward(self, predictions, targets,delta=1):
        flag = torch.abs(targets-predictions)
        loss = torch.where(
            flag < delta,
            ((targets-predictions)**2)/2,  # MSE part when the error is smaller than delta
            delta*((targets-predictions)-delta*0.5)  # MAE part for larger errors
        )
        return loss.mean()
        # if flag < delta:
        #     return ((targets-predictions)**2)/2
        # else:
        #     return delta*((targets-predictions)-delta*0.5)