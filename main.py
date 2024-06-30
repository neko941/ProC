import os
import sys
from utils import Options, SetSeed, BinanceDataset
import polars as pl
import torch
from losses import Evaluator
import glob

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
        path = os.path.join(args.save_path, str(t))
        os.makedirs(path, exist_ok=True)
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

        pl.from_dicts(history).write_csv(os.path.join(path, 'history.csv'))

    # Aggregate best metrics from all history files
    all_best_metrics = []
    for file_path in glob.glob(os.path.join(args.save_path, '**', 'history.csv'), recursive=True):
        df = pl.read_csv(file_path)
        best_metrics = {name: df[name].min() for name in evaluator.metrics}
        all_best_metrics.append(best_metrics)

    # Calculate mean and std for best metrics
    all_best_metrics_df = pl.DataFrame(all_best_metrics)
    summary_df = pl.DataFrame({
        "metric": evaluator.metrics.keys(),
        "mean": all_best_metrics_df.mean().row(0),
        "std": all_best_metrics_df.std().row(0)
    })

    summary_df.write_csv(os.path.join(args.save_path, 'summary_metrics.csv'))

    print(summary_df)