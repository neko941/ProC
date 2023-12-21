import numpy as np
import copy

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.00001):
        """
        Initializes the EarlyStopping object.

        Args:
        patience (int): Number of epochs to wait after improvement before stopping. Default is 7.
        verbose (bool): If True, prints a message for each validation loss improvement. Default is False.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = np.Inf  # Initialize best_score to a large value
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_weight = None

    def __call__(self, val_loss, weight):
        """
        Call this in each validation phase of training.

        Args:
        val_loss (float): The current validation loss.
        """
        if val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
            self.val_loss_min = val_loss
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            self.best_weight = copy.deepcopy(weight)
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
