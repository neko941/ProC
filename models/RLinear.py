import torch.nn as nn
from layers import RevIN

class RLinear(nn.Module):
    """
    Source: https://github.com/plumprc/RTSF/blob/main/models/RLinear.py
    """
    def __init__(self, configs):
        super(RLinear, self).__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.in_channels = configs.input_channels
        self.out_channels = configs.output_channels
        
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.rev = RevIN(self.in_channels)
        self.fc = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        # x: [B, seq_len, in_channels]
        x = self.rev(x, 'norm')
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = self.rev(x, 'denorm')
        x = self.fc(x)
        return x # [B, pred_len, out_channels]