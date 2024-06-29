import os
import sys

from utils import Options, SetSeed, BinanceDataset
from models import Linear
import torch.nn as nn
import torch

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)  # root directory
if ROOT not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH
ROOT = os.path.relpath(ROOT, os.getcwd())  # relative


if __name__ == "__main__":
    options = Options(ROOT)()
    args = options.args
    dataset = BinanceDataset(args).execute()
    options.add_args('input_channels', dataset.input_channels)
    options.add_args('output_channels', dataset.output_channels)

    for t in range(args.times):
        SetSeed(seed=args.seed+t).set()
        model = Linear(configs=args).to(args.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(args.epochs):
            losses = []
            for x, y in dataset.train_loader:
                x, y = x.to(args.device), y.to(args.device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {sum(losses)/len(losses)}")