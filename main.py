import argparse
import numpy as np
import os
import torch
import pandas as pd
import torch.nn as nn
from torch import optim

from dataloaders.provider import provider 
from models.architectures.LSTM import VanillaLSTM
from configs.model_configs import VanillaLSTMConfig
from utils.tools import EarlyStopping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./datasets")
    parser.add_argument("--data_path", type=str, default="AMZN.csv")
    parser.add_argument("--save_path", type=str, default="./checkpoints")

    parser.add_argument("--model", type=str, default="VanillaLSTM")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--features", type=str, default="S")
    parser.add_argument("--date", type=str, default="date")
    parser.add_argument("--target", type=str, default="OT")
    
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--scheduler", type=str, default="StepLR")

    class MainExp:
        def __init__(self, args):
            self.args = args
            self._build_model()
            
        def _build_model(self):
            model_dict = {
                'VanillaLSTM': VanillaLSTM,
            }
            config_dict = {
                'VanillaLSTM': VanillaLSTMConfig
            }

            self.model = model_dict[self.args.model](self.args, config_dict[self.args.model]())
            
        def _get_criterion(self):
            loss_dict = {
                "MSE": nn.MSELoss(),
            }
            return loss_dict[self.args.loss]
        
        def _get_optimizer(self):
            optimizer_dict = {
                "Adam": optim.Adam(self.model.parameters(), lr=self.args.learning_rate),
            }
            return optimizer_dict[self.args.optimizer]
        
        def _get_scheduler(self, optimizer):
            scheduler_dict = {
                "StepLR": optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
            }
            return scheduler_dict[self.args.scheduler]

        def train(self):
            train_loader = provider(self.args, "train")
            val_loader = provider(self.args, "val")
            optim = self._get_optimizer()
            scheduler = self._get_scheduler(optim)
            criterion = self._get_criterion()
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
            
            print("Start training...")
            for epoch in range(self.args.epochs):
                train_loss = []
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
                    scheduler.step()
                    
                    train_loss.append(loss.item())
                    print("Epoch: {}, Iteration: {}, Loss: {}".format(epoch+1, i, loss))
                train_loss = np.average(train_loss)
                val_loss = self.validate(val_loader, criterion)
                
                print("Epoch: {}, Train Loss: {}, Validate Loss: {}".format(epoch+1, train_loss, val_loss))
                early_stopping(val_loss, self.model, self.args.save_path)
        
        def validate(self, val_loader, criterion):
            self.model.eval()
            val_loss = []
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(val_loader):
                    batch_x = batch_x.float()
                    batch_x = batch_x.permute(0, 2, 1)
                    batch_y = batch_y.float()
                    batch_y = batch_y.permute(0, 2, 1)
                    output = self.model(batch_x)
                    
                    loss = criterion(output, batch_y)
                    val_loss.append(loss.item())
            val_loss = np.average(val_loss)
            self.model.train()
            return val_loss
        
        def test(self):
            pass
            
    args = parser.parse_args()
    exp = MainExp(args)
    exp.train()
