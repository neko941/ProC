import os
import json
import torch
import argparse
from .general import increment_path
from rich.table import Table
from rich import box
from rich.terminal_theme import MONOKAI
from rich.console import Console

class Options:
    def __init__(self, ROOT):
        self.ROOT = ROOT
    
    def parse_options(self):
        parser = argparse.ArgumentParser()

        # general 
        parser.add_argument('--times', type=int, default=1, help='number of times to run the experiment')
        parser.add_argument('--seed', type=int, default=941)
        parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"])
        parser.add_argument('--device_id', type=str, default="0")
        parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train the model')

        # save path
        parser.add_argument('--project', type=str, default=os.path.join(self.ROOT, 'runs'), help='project name')
        parser.add_argument('--name', type=str, default='exp', help='name of this experiment')
        parser.add_argument('--sep', type=str, default='', help='separator for name')

        # dataset
        parser.add_argument('--dataset', type=str, default='binance', help='dataset name')
        parser.add_argument('--input_len', type=int, default=10, help='input sequence length')
        parser.add_argument('--output_len', type=int, default=5, help='output sequence length')
        parser.add_argument('--split_ratio', type=float, default=[0.7, 0.2, 0.1], action='append', help='train, val, test split ratio')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--shuffle', action='store_true', help='shuffle dataset')

        self.args = parser.parse_args()
        return self

    def add_args(self, name, value):
        self.args.__dict__[name] = value

    def _fix_save_path(self):
        path = increment_path(
            os.path.join(
                self.args.project, 
                self.args.name
            ), 
            exist_ok=False, 
            sep=self.args.sep
        )
        self.add_args('save_path', path)
        if not os.path.exists(path):
            os.makedirs(path)
    
    def _fix_device(self):
        if self.args.device == 'cuda' and not torch.cuda.is_available():
            print("cuda is not available. Using cpu instead.")
            self.add_args('device', 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.device_id
    
    def fix_args(self):
        self._fix_save_path()
        self._fix_device()
        return self

    def save(self):
        """
        Saves all values in self.args to a file.
        """
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as f:
            json.dump(vars(self.args), f, indent=4)
    
    def display(self):
        """
        Displays all values in self.args using rich package.
        """

        table = Table(title="Experiment Arguments", box=box.ROUNDED)
        table.add_column("Argument", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for arg in vars(self.args):
            table.add_row(arg, str(getattr(self.args, arg)))

        console = Console(record=True)
        console.print(table)
        console.save_svg(os.path.join(self.args.save_path, 'configs.svg'), theme=MONOKAI)
    
    def __call__(self):
        self.parse_options()
        self.fix_args()
        self.display()
        self.save()
        return self