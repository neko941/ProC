import torch
import torch.nn as nn
from torch import optim
import numpy as np
from pathlib import Path
import time
import random
import math
from rich.layout import Layout
from rich.live import Live


from callbacks.early_stopping import EarlyStopping
from dataloaders.salinity import SalinityDSLoader
from dataloaders.provider import provider
from utils.visuals import progress_bar
from utils.visuals import table
from utils.metrics import metric

from models.architectures.LSTM import VanillaLSTM
from models.architectures.PatchTST import PatchTST
from models.architectures.DLinear import DLinear
from configs.model_configs import *

class AbstractAlgorithm:
    def __init__(self, opt):
        self.opt = opt
        self._set_seed()
        self._create_dir()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

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
            'PatchTST': PatchTST,
            'DLinear': DLinear,
        }
        config_dict = {
            'VanillaLSTM': VanillaLSTMConfig,
            'PatchTST': PatchTSTConfig,
            'DLinear': DLinearConfig
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

    def _get_scheduler(self, optimizer, training_steps=-1, steps_per_epoch=10, ):
        scheduler_dict = {
            "StepLR": optim.lr_scheduler.StepLR,
            "LambdaLR": optim.lr_scheduler.LambdaLR,
            "OneCycleLR": optim.lr_scheduler.OneCycleLR,
        }
        if self.opt.scheduler == 'LambdaLR':
            warmup_steps = self.opt.warmup_steps if self.opt.warmup_steps > 0 else math.ceil(self.opt.warmup_ratio * training_steps)
            lr_lambda = lambda step: 1.0 if step >= warmup_steps else float(step) / float(max(1, warmup_steps))
            return scheduler_dict[self.opt.scheduler](optimizer, lr_lambda=lr_lambda)
        elif self.opt.scheduler == 'OneCycleLR':
            return scheduler_dict[self.opt.scheduler](optimizer, steps_per_epoch=steps_per_epoch, pct_start=self.opt.pct_start, epochs=self.opt.epochs, max_lr=self.opt.learning_rate)
        else:
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
        criterion = self._get_criterion()
        optimizer = self._get_optimizer()

        early_stopping = EarlyStopping(patience=self.opt.patience, verbose=False)
        train_loader = provider(self.opt, flag='train')
        samples = len(train_loader)
        scheduler = self._get_scheduler(optimizer=optimizer, steps_per_epoch=samples)

        layout = Layout()
        progress = progress_bar()
        tab = table(columns=['Epoch', 'Loss', 'Val Loss', 'Patience', 'Time'])

        layout.split(
            Layout(name="upper", size=3),
            Layout(name="lower")
        )
        layout["upper"].update(progress)
        layout["lower"].update(tab)

        with Live(layout, refresh_per_second=4, vertical_overflow='visible'):
            overall_task = progress.add_task("[green]Epoch", total=self.opt.epochs)
            subtask = progress.add_task("[red]Batch", total=samples)
            
            for epoch in range(self.opt.epochs):
                progress.reset(subtask, total=samples)
                running_loss = []
                start_inner = time.time()
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                    # batch_x = batch_x.permute(0, 2, 1) # batch, channels, length

                    outputs = self.model(batch_x)
                    dims = 0
                    
                    outputs = outputs[:, -self.opt.prediction_length:, dims:]
                    batch_y = batch_y[:, -self.opt.prediction_length:, dims:]

                    loss = criterion(outputs, batch_y)
                    running_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    t = time.time() - start_inner
                    progress.update(subtask, advance=1, description=f'[red]Batch | loss: {loss.item():.4f}')

                running_loss = np.array(running_loss)
                average_loss = running_loss.sum() / samples
                progress.update(overall_task, advance=1, description=f'[green]Epoch | loss: {average_loss:.4f}')

                val_loss = self.evaluate()[0]
                early_stopping(val_loss, weight=self.model.state_dict())
                tab.add_row(f"{epoch + 1}", 
                            f"{average_loss:.4f}", 
                            f"{val_loss:.4f}", 
                            f"{self.opt.patience-early_stopping.counter}", 
                            f"{t:.4f}")

                # Check for early stopping
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    # print("Loading the best weights...")
                    # self.model.load_state_dict(early_stopping.best_weight)
                    break

    
    # def train_epoch(self, train_loader, optimizer, criterion, progress, subtask):
        
    #     return progress, np.array(running_loss), time.time()-start

    def evaluate(self, flag='val'):
        self.model.eval()  # Set the model to evaluation mode

        # test_loader = self._get_loader(type='test')
        if flag == 'test': loader = provider(self.opt, flag=flag)
        else: loader = provider(self.opt, flag=flag)
        
        total_loss = 0.0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                # batch_x = batch_x.permute(0, 2, 1)

                outputs = self.model(batch_x)
                dims = 0
                
                outputs = outputs[:, -self.opt.prediction_length:, dims:]
                batch_y = batch_y[:, -self.opt.prediction_length:, dims:]

                criterion = self._get_criterion()
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y.cpu().numpy())

        average_loss = total_loss / len(loader)
        print(f'Loss: {average_loss:.4f}')

        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mae, mse, rmse, mape, mspe, rse, corr, r2 = metric(predictions, targets)
        
        return average_loss, r2


        
    