import torch
import torch.nn as nn
from torch import optim
from models.architectures.LSTM import VanillaLSTM
from configs.model_configs import VanillaLSTMConfig
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
from callbacks.early_stopping import EarlyStopping
from sklearn.metrics import r2_score

class AbstractAlgorithm:
    model_dict = {
        'VanillaLSTM': VanillaLSTM,
    }
    config_dict = {
        'VanillaLSTM': VanillaLSTMConfig,
    }
    loss_dict = {
        "MSE": nn.MSELoss,
    }
    optimizer_dict = {
        "Adam": optim.Adam,
    }
    scheduler_dict = {
        "StepLR": optim.lr_scheduler.StepLR,
    }

    class DataGenerator(Dataset):
        def __init__(self, X, y):
            self.X = torch.Tensor(X)
            self.y = torch.Tensor(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            return self.X[index], self.y[index]

    def preprocessing(self, 
                      x, 
                      y, 
                      batchsz:int):
        return DataLoader(self.DataGenerator(x, y), batch_size=batchsz, shuffle=True)
    
    def __init__(self, opt, save_dir):
        self.opt = opt
        self.path_weight = Path(save_dir) / opt.model /  'weights'
        self.path_weight.mkdir(parents=True, exist_ok=True)
    
    def train(self, xtrain, ytrain, xval, yval, extension='.pt'):
        start = time.time()
        if extension != '.pt':
            best = Path(self.path_weight, self.__class__.__name__, 'best')
            last = Path(self.path_weight, self.__class__.__name__, 'last')
        else:
            best = Path(self.path_weight, self.__class__.__name__)
            last = best
        best.mkdir(parents=True, exist_ok=True)
        last.mkdir(parents=True, exist_ok=True)

        configs = AbstractAlgorithm.config_dict[self.opt.model]()
        self.model = AbstractAlgorithm.model_dict[self.opt.model](self.opt, configs)
        if configs.weight_path is not None:
            self.model.load_state_dict(torch.load(configs.weight_path))
        criterion = AbstractAlgorithm.loss_dict[self.opt.loss]()
        optimizer = AbstractAlgorithm.optimizer_dict[self.opt.optimizer](self.model.parameters(),lr=self.opt.learning_rate)
        scheduler = AbstractAlgorithm.scheduler_dict[self.opt.scheduler](optimizer, step_size=10, gamma=0.1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        early_stopping = EarlyStopping(patience=self.opt.patience, verbose=True)
        train_loader = self.preprocessing(x=xtrain, y=ytrain, batchsz=self.opt.batch_size)
        val_loader = self.preprocessing(x=xval, y=yval, batchsz=self.opt.batch_size)
        self.time_used = time.time() - start

        for epoch in range(self.opt.epochs):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_x = batch_x.permute(0, 2, 1)

                outputs = self.model(batch_x)
                dims = -1
                outputs = outputs[:, dims:, :].squeeze()

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.opt.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


    def evaluate(self, x, y):
        self.model.eval()  # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        val_loader = self.preprocessing(x=x, y=y, batchsz=self.opt.batch_size)
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_x = batch_x.permute(0, 2, 1)

                outputs = self.model(batch_x)
                dims = -1
                outputs = outputs[:, dims:, :].squeeze()

                criterion = AbstractAlgorithm.loss_dict[self.opt.loss]()
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        average_loss = total_loss / len(val_loader)
        print(f'Loss: {average_loss:.4f}')

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        r2 = r2_score(all_targets, all_predictions)
        print(f'R-squared (R2) Score: {r2:.4f}')

        return average_loss, r2


        
    