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
from utils.general import set_seed 
from utils.general import list_uniqifier 
from dataloaders.loaders import SalinityDataset
from algorithms.abstract import AbstractAlgorithm

def main(opt):
    save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, mkdir=True))
    set_seed(opt.seed)
    dataset = SalinityDataset(low_memory=opt.low_memory, normalization=opt.normalization)
    xtrain, ytrain, xval, yval, xtest, ytest = dataset(save_dir=save_dir,
                                                       split_ratio=(opt.train_size, opt.val_size, 1-opt.train_size-opt.val_size),
                                                       lag=opt.sequence_length,
                                                       ahead=opt.prediction_length,
                                                       offset=opt.offset)
    print(f'{xtrain.shape = }')
    print(f'{ytrain.shape = }')
    print(f'{xval.shape = }')
    print(f'{yval.shape = }')
    print(f'{xtest.shape = }')
    print(f'{ytest.shape = }')
    
    for model in list_uniqifier(opt.models):
        opt.model = model
        algo = AbstractAlgorithm(opt, save_dir=save_dir)
        algo.train(xtrain, ytrain, xval, yval)
        algo.evaluate(xtest, ytest)
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