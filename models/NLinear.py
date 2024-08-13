import torch.nn as nn

class NLinear(nn.Module):
    """
    Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py
    """
    def __init__(self, configs):
        super(NLinear, self).__init__()
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len
        self.in_channels = configs.input_channels
        self.out_channels = configs.output_channels
        
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.fc = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        # x: [B, seq_len, in_channels]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        x = self.fc(x)
        return x # [B, pred_len, out_channels]