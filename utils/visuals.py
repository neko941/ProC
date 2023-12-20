# import os
# import matplotlib.pyplot as plt

from rich.progress import Progress
from rich.progress import BarColumn 
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import MofNCompleteColumn
from rich.progress import TimeRemainingColumn
from rich.table import Table
from rich import box as rbox
# def save_plot(filename, data, xlabel=None, ylabel=None):
#     fig, ax = plt.subplots()
#     for datum in data: ax.plot(*datum['data'], color=datum['color'], label=datum['label'])
#     ax.legend()
#     if xlabel is not None: ax.set_xlabel(xlabel)
#     if ylabel is not None: ax.set_ylabel(ylabel)
#     plt.title(os.path.basename(filename).split('.')[0])
#     fig.savefig(filename)
#     plt.close('all')

def progress_bar():
    return Progress("[bright_cyan][progress.description]{task.description}",
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("•Items"),
                    MofNCompleteColumn(), # "{task.completed}/{task.total}",
                    TextColumn("•Remaining"),
                    TimeRemainingColumn(),
                    TextColumn("•Total"),
                    TimeElapsedColumn())

def table(columns):
    table = Table(title="[cyan]Results", show_header=True, header_style="bold magenta", box=rbox.ROUNDED, show_lines=True)
    [table.add_column(f'[green]{name}', justify='center') for name in columns]

    return table