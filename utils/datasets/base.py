from abc import ABC, abstractmethod
import polars as pl
from torch.utils.data import DataLoader, TensorDataset
import torch

class DatasetController(ABC):
    def __init__(self, configs):
        self.split_ratio = configs.split_ratio
        self.batch_size = configs.batch_size
        self.shuffle = configs.shuffle

    def read(self):
        self.df = pl.read_csv(self.data_paths)
    
    def __len__(self):
        return len(self.df)

    @abstractmethod
    def split(self):
        pass

    def preprocessing(self):
        self.df = self.df.shrink_to_fit()

    def validate(self):
        assert sum(self.split_ratio) <= 1, 'Split ratio must be less than or equal 1'

    def get_loader(self):
        self.train_loader = self._get_train_loader()
        self.val_loader = self._get_val_loader()
        self.test_loader = self._get_test_loader()

    def _get_train_loader(self):
        return DataLoader(
            dataset=TensorDataset(torch.tensor(self.x_train, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.float32)),
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        )

    def _get_val_loader(self):
        return DataLoader(
            dataset=TensorDataset(torch.tensor(self.x_val, dtype=torch.float32), torch.tensor(self.y_val, dtype=torch.float32)), 
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        )
    
    def _get_test_loader(self):
        return DataLoader(
            dataset=TensorDataset(torch.tensor(self.x_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.float32)), 
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        )
    
    def execute(self):
        self.validate()
        self.read()
        self.preprocessing()
        self.split()
        self.get_loader()
        return self

