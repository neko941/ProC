import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.options import parse_opt
from utils.general import increment_path 
from utils.general import list_uniqifier 
from algorithms.abstract import AbstractAlgorithm

def main(opt):
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, mkdir=True))
    
    for model in list_uniqifier(opt.models):
        opt.model = model
        algo = AbstractAlgorithm(opt)
        algo.train()
        print('Evaluating...')
        algo.evaluate()
    return

def run(**kwargs):
    """ 
    Usage (example)
        import train
        train.run(all=True, 
                  configsPath=data.yaml,
                  lag=5,
                  ahead=1,
                  offset=1)
    """
    opt = parse_opt(ROOT=ROOT)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt(ROOT=ROOT)
    main(opt)