import torch
import torch.nn as nn
from torch import optim
from models.architectures.LSTM import VanillaLSTM
from models.architectures.PatchTST import PatchTST
from configs.model_configs import VanillaLSTMConfig, PatchTSTConfig
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import time
import random
import math
from callbacks.early_stopping import EarlyStopping
from dataloaders.salinity import SalinityDSLoader
from dataloaders.provider import provider
from sklearn.metrics import r2_score
from utils.visuals import progress_bar

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
            'PatchTST': PatchTST,
        }
        config_dict = {
            'VanillaLSTM': VanillaLSTMConfig,
            'PatchTST': PatchTSTConfig,
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

    def _get_scheduler(self, optimizer, training_steps=-1):
        scheduler_dict = {
            "StepLR": optim.lr_scheduler.StepLR,
            "LambdaLR": optim.lr_scheduler.LambdaLR,
        }
        if self.opt.scheduler == 'LambdaLR':
            warmup_steps = self.opt.warmup_steps if self.opt.warmup_steps > 0 else math.ceil(self.opt.warmup_ratio * training_steps)
            lr_lambda = lambda step: 1.0 if step >= warmup_steps else float(step) / float(max(1, warmup_steps))
            return scheduler_dict[self.opt.scheduler](optimizer, lr_lambda=lr_lambda)

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
        # start = time.time()

        criterion = self._get_criterion()
        optimizer = self._get_optimizer()
        # scheduler = self._get_scheduler(optimizer=optimizer)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # early_stopping = EarlyStopping(patience=self.opt.patience, verbose=True)
        # train_loader = self._get_loader(type='train')
        # val_loader = self._get_loader(type='val')
        train_loader = provider(self.opt, flag='train')
        # val_loader = provider(self.opt, flag='val')
        # self.time_used = time.time() - start

        from rich.table import Table
        from rich.console import Console
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Epoch", style="dim", width=12)
        table.add_column("Average Loss", justify="right")

        with progress_bar() as progress:
            samples = len(train_loader)
            overall_task = progress.add_task("[green]Epochs", total=self.opt.epochs)
            subtask = progress.add_task("[red]Step", total=samples)
            for _ in range(self.opt.epochs):
                running_loss = 0.0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
                    batch_x = batch_x.permute(0, 2, 1)

                    outputs = self.model(batch_x)
                    dims = 0
                    
                    outputs = outputs[:, -self.opt.prediction_length:, dims:]
                    batch_y = batch_y[:, -self.opt.prediction_length:, dims:]

                    loss = criterion(outputs, batch_y)
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    progress.update(subtask, advance=1, description=f'[red]Step   | loss: {loss.item():.4f}')
                progress.reset(subtask, total=samples)
                progress.update(overall_task, advance=1, description=f'[green]Epochs | loss: {running_loss / samples:.4f}')
        
    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # test_loader = self._get_loader(type='test')
        test_loader = provider(self.opt, flag='test')
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
                batch_x = batch_x.permute(0, 2, 1)

                outputs = self.model(batch_x)
                dims = 0
                
                outputs = outputs[:, -self.opt.prediction_length:, dims:]
                batch_y = batch_y[:, -self.opt.prediction_length:, dims:]

                criterion = self._get_criterion()
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        average_loss = total_loss / len(test_loader)
        print(f'Loss: {average_loss:.4f}')

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        all_predictions = np.reshape(all_predictions, (all_predictions.shape[0] * all_predictions.shape[1], -1))
        all_targets = np.reshape(all_targets, (all_targets.shape[0] * all_targets.shape[1], -1))

        r2 = r2_score(all_targets, all_predictions)
        print(f'R-squared (R2) Score: {r2:.4f}')

        return average_loss, r2


        
    