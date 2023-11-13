import pandas as pd
import os

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class GeneralDataset(Dataset):
        def __init__(
            self,
            root_path,
            seq_len,
            label_len,
            pred_len,
            flag,
            data_path="AMZN.csv",
            features="S",
            date="date",
            target="OT",
            scale=True,
        ):
            self.root_path = root_path
            self.data_path = data_path

            assert flag in ["train", "val", "test"]
            type_mapping = {"train": 0, "val": 1, "test": 2}
            self.type = type_mapping[flag]
            
            self.seq_len = seq_len
            self.label_len = label_len
            self.pred_len = pred_len
            self.features = features
            self.date = date
            self.target = target
            self.scale = scale
            self.__read_data__()

        def __read_data__(self):
            self.scaler = StandardScaler()
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            df_len = len(df_raw)

            num_train = int(df_len * 0.7)
            num_val = int(df_len * 0.2)
            num_test = df_len - num_train - num_val

            border1s = [0, num_train - self.seq_len, df_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_val, df_len]
            border1 = border1s[self.type]
            border2 = border2s[self.type]

            columns = list(df_raw.columns)
            columns.remove(self.date)
            columns.remove(self.target)
            df_raw = df_raw[[self.date] + columns + [self.target]]
            
            if self.features == 'M' or self.features == 'MS':
                data = df_raw[columns[1:]]
            else:
                data = df_raw[[self.target]]
                
            if self.scale:
                self.scaler.fit(data[border1s[0]:border2s[0]].values)
                data = self.scaler.transform(data.values)
            else:
                data = data.values
            
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

        def __getitem__(self, index):
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            return seq_x, seq_y

        def __len__(self):
            return len(self.data_x) - self.seq_len - self.pred_len + 1