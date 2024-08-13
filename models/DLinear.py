import torch
import torch.nn as nn

class DLinear(nn.Module):
    """
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
    """
    def __init__(self, configs):
        super(DLinear, self).__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.in_channels = configs.input_channels
        self.out_channels = configs.output_channels
        
        self.kernel_size = 25
        self.decompsition = SeriesDecomposition(self.kernel_size)
        
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.fc = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        # x: [B, seq_len, in_channels]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0,2,1)

        x = self.fc(x)
        return x # [B, pred_len, out_channels]

class MovingAverage(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomposition(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean