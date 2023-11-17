import os
from .abstract import AbstractLoader
    
class SalinityDataset(AbstractLoader):
    def __init__(self, low_memory, normalization):
        super(SalinityDataset, self).__init__(low_memory=low_memory, normalization=normalization)
        self.path = os.path.join('datasets', 'Salinity', 'processed', 'by_station-mekong')
        self.target_features = ['average']
        self.date_feature = 'date'
        self.train_features = ['station']
        self.use_target_features = True
        self.granularity = 1440 # minutes
        self.keys = ['station']
        self.file_name_as_feature:str = 'station'