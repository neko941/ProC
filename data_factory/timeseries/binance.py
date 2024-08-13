from .base import TimeSeriesDataset
import polars as pl

class BinanceDataset(TimeSeriesDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_paths = '.\datasets\CryptoDownload\Binance\Binance_BNBUSDT_d.csv'
        self.input_features = ['Open', 'High', 'Low', 'Close']
        self.output_features = ['Close']
        self.input_channels = len(self.input_features)
        self.output_channels = len(self.output_features)
    
    def read(self):
        self.df = pl.read_csv(self.data_paths, skip_rows=1)