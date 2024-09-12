import torch.nn as nn
from losses import rmseloss
MSE = nn.MSELoss
MAE = nn.L1Loss
RMSE = rmseloss.RMSELoss

# Automatically create a list of all classes imported in this file
import sys
import inspect
LOSSES = [name for name, obj in sys.modules[__name__].__dict__.items() if inspect.isclass(obj)]
print(f'{LOSSES = }')

class Evaluator:
    def __init__(self):
        self.metrics = {
            'MAE': MAE(),
            'MSE': MSE(),
            'RMSE': RMSE()
        }

    def evaluate(self, y_pred, y):
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(y_pred, y).item()
        return results
