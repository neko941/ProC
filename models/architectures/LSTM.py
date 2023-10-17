import torch.nn as nn

class VanillaLSTM(nn.Module):
    def __init__(self, input_size, units, output_size):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, units[0])  
        self.fc1 = nn.Linear(units[0], units[1])  
        self.fc2 = nn.Linear(units[1], output_size)  

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc1(lstm_out[:, -1, :])  
        output = self.fc2(output)
        return output
