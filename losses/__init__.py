import torch.nn as nn
MSE = nn.MSELoss
MAE = nn.L1Loss

class Evaluator:
    def __init__(self):
        self.metrics = {
            'MAE': MAE(),
            'MSE': MSE(),
        }

    def evaluate(self, y_pred, y):
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(y_pred, y).item()
        return results