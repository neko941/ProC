import argparse
import os
import pandas as pd
import torch.nn as nn
from torch import optim
from models.architectures import LSTM
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./datasets")
    parser.add_argument("--data_path", type=str, default="AMZN.csv")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)

    class CustomDataset(Dataset):
        def __init__(
            self,
            root_path,
            seq_len,
            label_len,
            pred_len,
            data_path="AMZN.csv",
            # features="S",
        ):
            self.root_path = root_path
            self.data_path = data_path

            # self.features = features

            self.seq_len = seq_len
            self.label_len = label_len
            self.pred_len = pred_len
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
            border1 = border1s[0]
            border2 = border2s[0]

            columns = df_raw.columns
            # data = df_raw[columns[1:]].values
            data = df_raw[['High']].values
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

            # self.scaler.fit(self.data_x)
            # self.data_x = self.scaler.transform(self.data_x)

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

    def data_loader(args):
        print("Loading dataset...")
        dataset = CustomDataset(
            root_path=args.root_path,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            data_path=args.data_path,
        )

        train_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

        return train_loader

    class MainExp:
        def __init__(self, args):
            self.args = args
            self.model = LSTM.VanillaLSTM(
                input_size=args.seq_len,
                units=[64, 64],
                output_size=args.pred_len,
            )
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        def train(self):
            train_loader = data_loader(self.args)
            optim = self.optimizer
            criterion = self.criterion
            print("Start training...")
            for epoch in range(self.args.epochs):
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    optim.zero_grad()
                    batch_x = batch_x.float()
                    batch_x = batch_x.permute(0, 2, 1)
                    batch_y = batch_y.float()
                    batch_y = batch_y.permute(0, 2, 1)
                    output = self.model(batch_x)

                    loss = criterion(output, batch_y)
                    loss.backward()
                    optim.step()
                    print("Epoch: {}, Iteration: {}, Loss: {}".format(epoch, i, loss))
            
    args = parser.parse_args()
    exp = MainExp(args)
    exp.train()
