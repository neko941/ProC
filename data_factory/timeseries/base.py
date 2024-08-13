from ..base import DatasetController
import numpy as np

class TimeSeriesDataset(DatasetController):
    def __init__(self, configs):
        super().__init__(configs)
        self.seq_len = configs.input_len
        self.pred_len = configs.output_len

    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1
    
    def split(self):
        x, y = [], []

        for i in range(self.__len__()):
            # Extract sequences
            seq_x = self.df[i : i + self.seq_len][self.input_features]
            seq_y = self.df[i + self.seq_len : i + self.seq_len + self.pred_len][self.output_features]

            # Check for null values
            if sum(seq_x.null_count().to_numpy()[0])!=0 or sum(seq_y.null_count().to_numpy()[0])!=0:
                continue  # Skip sequences with null values

            # Convert to NumPy arrays
            seq_x = seq_x.to_numpy()
            seq_y = seq_y.to_numpy()

            x.append(seq_x)
            y.append(seq_y)

        # split the data into training, validation, and test sets
        if self.split_ratio is not None:
            self.x_train, self.x_val, self.x_test = np.split(x, [int(self.split_ratio[0] * len(x)), int((self.split_ratio[0] + self.split_ratio[1]) * len(x))])
            self.y_train, self.y_val, self.y_test = np.split(y, [int(self.split_ratio[0] * len(y)), int((self.split_ratio[0] + self.split_ratio[1]) * len(y))])
            print(f'{self.x_train.shape = }')
            print(f'{self.y_train.shape = }')
            print(f'{self.x_val.shape = }')
            print(f'{self.y_val.shape = }')
            print(f'{self.x_test.shape = }')
            print(f'{self.y_test.shape = }')
    