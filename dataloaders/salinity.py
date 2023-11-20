import torch
from torch.utils.data import Dataset, DataLoader

from dataloaders.loaders import SalinityDataset

class DataGenerator(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
            return self.X[index], self.y[index]

class SalinityDSLoader:
    def __init__(self, opt):
        self.opt = opt
        self._get_data()
        
    def _get_data(self):
        dataset = SalinityDataset(low_memory=self.opt.low_memory, normalization=self.opt.normalization)
        self.xtrain, self.ytrain, self.xval, self.yval, self.xtest, self.ytest = dataset(save_dir=self.opt.save_dir,
                                                split_ratio=(self.opt.train_size, self.opt.val_size, 1-self.opt.train_size-self.opt.val_size),
                                                lag=self.opt.sequence_length,
                                                ahead=self.opt.prediction_length,
                                                offset=self.opt.offset)
        print(f'{self.xtrain.shape = }')
        print(f'{self.ytrain.shape = }')
        print(f'{self.xval.shape = }')
        print(f'{self.yval.shape = }')
        print(f'{self.xtest.shape = }')
        print(f'{self.ytest.shape = }')
        
    def preprocessing(self, x, y):
        return DataLoader(DataGenerator(x, y), batch_size=self.opt.batch_size, shuffle=True)
    
    def _get_loader(self, type='train'):
        if type == 'train':
            return self.preprocessing(self.xtrain, self.ytrain)
        elif type == 'val':
            return self.preprocessing(self.xval, self.yval)
        else:
            return self.preprocessing(self.xtest, self.ytest)
    
    
