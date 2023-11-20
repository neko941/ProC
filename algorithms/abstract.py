import torch
import torch.nn as nn
from torch import optim
from models.architectures.LSTM import VanillaLSTM
from configs.model_configs import VanillaLSTMConfig
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import time
import random
from callbacks.early_stopping import EarlyStopping
from dataloaders.salinity import SalinityDSLoader

class AbstractAlgorithm:
    def __init__(self, opt):
        self.opt = opt
        self._set_seed()
        self._create_dir()
        self.model = self._build_model()

    def _create_dir(self):
        self.path_weight = Path(self.opt.save_dir) / self.opt.model /  'weights'
        self.path_weight.mkdir(parents=True, exist_ok=True)
        
        if self.opt.extension != '.pt':
            best = Path(self.path_weight, self.__class__.__name__, 'best')
            last = Path(self.path_weight, self.__class__.__name__, 'last')
        else:
            best = Path(self.path_weight, self.__class__.__name__)
            last = best
        best.mkdir(parents=True, exist_ok=True)
        last.mkdir(parents=True, exist_ok=True)
        
    def _build_model(self):
        model_dict = {
            'VanillaLSTM': VanillaLSTM,
        }
        config_dict = {
            'VanillaLSTM': VanillaLSTMConfig,
        }
        configs = config_dict[self.opt.model]()
        model = model_dict[self.opt.model](self.opt, configs)
        if configs.weight_path is not None:
            model.load_state_dict(torch.load(configs.weight_path))
        return model
    
    def _get_criterion(self):
        loss_dict = {
            "MSE": nn.MSELoss,
        }
        return loss_dict[self.opt.loss]()
    
    def _get_optimizer(self):
        optimizer_dict = {
            "Adam": optim.Adam,
        }
        return optimizer_dict[self.opt.optimizer](self.model.parameters(),lr=self.opt.learning_rate)

    def _get_scheduler(self, optimizer):
        scheduler_dict = {
            "StepLR": optim.lr_scheduler.StepLR,
        }
        return scheduler_dict[self.opt.scheduler](optimizer, step_size=10, gamma=0.1)
    
    def _get_loader(self, type='train'):
        loader_dict = {
            'SalinityDSLoader': SalinityDSLoader,
        }
        loader = loader_dict[self.opt.loader](self.opt)
        return loader._get_loader(type=type)
    
    def _set_seed(self):
        np.random.seed(self.opt.seed)
        random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed(self.opt.seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        # os.environ["PYTHONHASHSEED"] = str(self.opt.seed)
        print(f"Random seed set as {self.opt.seed}")
    
    def train(self):
        start = time.time()

        criterion = self._get_criterion()
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer=optimizer)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        early_stopping = EarlyStopping(patience=self.opt.patience, verbose=True)
        train_loader = self._get_loader(type='train')
        val_loader = self._get_loader(type='val')
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

        
    