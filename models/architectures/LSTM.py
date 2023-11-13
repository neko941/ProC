import torch.nn as nn

class VanillaLSTM(nn.Module):
    def __init__(self, args, configs):
        super(VanillaLSTM, self).__init__()
        input_size = args.seq_len
        units = configs.units
        output_size = args.pred_len

        self.lstm = nn.LSTM(input_size, units[0])  
        self.fc1 = nn.Linear(units[0], units[1])  
        self.fc2 = nn.Linear(units[1], output_size)  

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc1(lstm_out)
        output = self.fc2(output)
        return output
