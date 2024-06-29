import torch.nn as nn

class Linear(nn.Module):
    """
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/Linear.py
    """
    def __init__(self, configs):
        super(Linear, self).__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.in_channels = configs.input_channels
        self.out_channels = configs.output_channels
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        if self.in_channels != self.out_channels:
            self.channel_linear = nn.Linear(self.in_channels, self.out_channels)
        else:
            self.channel_linear = None

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        if self.channel_linear is not None:
            x = self.channel_linear(x)
        return x # [Batch, Output length, Channel]