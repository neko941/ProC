import torch.nn as nn
from losses import Huber, LogCosh, MAE, MAPE, MBE, MSE, MSLE, NRMSE, Quantile, RAE, RMSE, RMSLE, RRMSE, RSE

MSE = MSE.MSELoss
MAE = MAE.MAELoss
RMSE = RMSE.RMSELoss
Huber = Huber.HuberLoss
LogCosh = LogCosh.LogCoshLoss
MAPE = MAPE.MAPELoss
MBE = MBE.MBELoss
MSLE = MSLE.MSLELoss
NRMSE = NRMSE.NRMSELoss
Quantile = Quantile.QuantileLoss
RAE = RAE.RAELoss
RMSLE = RMSLE.RMSLELoss
RRMSE = RRMSE.RRMSELoss
RSE = RSE.RSELoss

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
            'RMSE':RMSE(),
            'Huber' : Huber(),
            'LogCosh' : LogCosh(),
            'MAPE' : MAPE(),
            'MBE' : MBE(),
            'MSLE' : MSLE(),
            'NRMSE' : NRMSE(),
            'Quantile' : Quantile(),
            'RAE' : RAE(),
            'RMSLE' : RMSLE(),
            'RRMSE' : RRMSE(),
            'RSE' : RSE()
        }

    def evaluate(self, y_pred, y):
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric(y_pred, y).item()
        return results