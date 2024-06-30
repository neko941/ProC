import os
import sys
from utils import Options, SetSeed, BinanceDataset
from models import Linear
import torch
from losses import Evaluator

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)  # root directory
if ROOT not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH
ROOT = os.path.relpath(ROOT, os.getcwd())  # relative

if __name__ == "__main__":
    options = Options(ROOT)()
    args = options.args
    dataset = BinanceDataset(args).execute()
    options.add_args('input_channels', dataset.input_channels)
    options.add_args('output_channels', dataset.output_channels)

    for t in range(args.times):
        SetSeed(seed=args.seed+t).set()
        model = getattr(__import__('models'), args.model)(configs=args).to(args.device)
        criterion = getattr(__import__('losses'), args.loss)()
        optimizer = getattr(__import__('optimizers'), args.optimizer)(model.parameters(), lr=args.lr)
        evaluator = Evaluator()

        history = []

        for epoch in range(args.epochs):
            model.train()
            train_losses = []
            for x, y in dataset.train_loader:
                x, y = x.to(args.device), y.to(args.device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                train_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_train_loss = sum(train_losses) / len(train_losses)

            # Evaluation on test set
            model.eval()
            test_results = {name: [] for name in evaluator.metrics}
            with torch.no_grad():
                for x, y in dataset.test_loader:
                    x, y = x.to(args.device), y.to(args.device)
                    y_pred = model(x)
                    results = evaluator.evaluate(y_pred, y)
                    for name, value in results.items():
                        test_results[name].append(value)

            for name in test_results:
                test_results[name] = sum(test_results[name]) / len(test_results[name])
            
            history.append({
                'epoch': epoch+1,
                'train_loss': avg_train_loss,
                **test_results,
            })

            print(history[-1])