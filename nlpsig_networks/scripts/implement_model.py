from __future__ import annotations

from nlpsig.classification_utils import DataSplits, Folds
from nlpsig_networks.pytorch_utils import _get_timestamp, training_pytorch, testing_pytorch, KFold_pytorch
from nlpsig_networks.focal_loss import FocalLoss
from typing import Iterable
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os


def implement_model(
    model: nn.Module,
    num_epochs: int,
    x_data: torch.tensor | np.array,
    y_data: torch.tensor | np.array,
    learning_rate: float,
    seed: int,
    loss: str,
    gamma: float = 0.0,
    device: str | None = None,
    batch_size: int = 64,
    data_split_seed: int = 0,
    split_ids: torch.Tensor | None = None,
    split_indices: tuple[Iterable[int], Iterable[int], Iterable[int]] | None = None,
    k_fold: bool = False,
    n_splits: int = 5,
    patience: int = 10,
    verbose_training: bool = True,
    verbose_results: bool = True,
):
    # set some variables for training
    return_best = True
    early_stopping = True
    model_output = f"best_model_{_get_timestamp()}.pkl"
    validation_metric = "f1"
    weight_decay_adam = 0.0001
    
    if device is not None:
        # set model to device is passed
        model.to(device)
        
        # convert data to tensors if passed as numpy arrays
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        if isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)
        
        # set data to device    
        x_data = x_data.to(device)
        y_data = y_data.to(device)

    if k_fold:
        # perform KFold evaluation and return the performance on validation and test sets
        # split dataset
        folds = Folds(x_data=x_data,
                      y_data=y_data,
                      groups=split_ids,
                      n_splits=n_splits,
                      indices=split_indices,
                      shuffle=True,
                      random_state=data_split_seed)
        
         # define loss
        if loss == "focal":
            criterion = FocalLoss(gamma = gamma)
        elif loss == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("criterion must be either 'focal' or 'cross_entropy'")

        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay_adam)
        
        # perform k-fold evaluation which returns a dataframe with columns for the
        # loss, accuracy, f1 (macro) and individual f1-scores for each fold
        # (for both validation and test set)
        results = KFold_pytorch(folds=folds,
                                model=model,
                                criterion=criterion,
                                optimizer=optimizer,
                                num_epochs=num_epochs,
                                batch_size=batch_size,
                                seed=seed,
                                return_best=return_best,
                                early_stopping=early_stopping,
                                patience=patience,
                                verbose=verbose_training)
    else:
        # split dataset
        data_loader_args = {"batch_size": batch_size, "shuffle": True}

        split_data = DataSplits(x_data=x_data,
                                y_data=y_data,
                                groups=split_ids,
                                train_size=0.8,
                                valid_size=0.2,
                                indices=split_indices,
                                shuffle=True,
                                random_state=data_split_seed)
        train, valid, test = split_data.get_splits(as_DataLoader=True,
                                                   data_loader_args=data_loader_args)

        # define loss
        if loss == "focal":
            criterion = FocalLoss(gamma = gamma)
            y_train = split_data.get_splits(as_DataLoader=False)[1]
            criterion.set_alpha_from_y(y=y_train)
        elif loss == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("criterion must be either 'focal' or 'cross_entropy'")

        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay_adam)
        
        # train model
        model = training_pytorch(model=model,
                                 train_loader=train,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 num_epochs=num_epochs,
                                 valid_loader=valid,
                                 seed=seed,
                                 return_best=return_best,
                                 output=model_output,
                                 early_stopping=early_stopping,
                                 validation_metric=validation_metric,
                                 patience=patience,
                                 verbose=verbose_training)
        
        # evaluate on validation
        valid_results = testing_pytorch(model=model,
                                        test_loader=valid,
                                        criterion=criterion,
                                        verbose=False)
        
        # evaluate on test
        test_results = testing_pytorch(model=model,
                                       test_loader=test,
                                       criterion=criterion,
                                       verbose=False)
        
        results = pd.DataFrame({"loss": test_results["loss"],
                                "accuracy": test_results["accuracy"], 
                                "f1": test_results["f1"],
                                "f1_scores": [test_results["f1_scores"]],
                                "precision": test_results["precision"],
                                "precision_scores": [test_results["precision_scores"]],
                                "recall": test_results["recall"],
                                "recall_scores": [test_results["recall_scores"]],
                                "valid_loss": valid_results["loss"],
                                "valid_accuracy": valid_results["accuracy"], 
                                "valid_f1": valid_results["f1"],
                                "valid_f1_scores": [valid_results["f1_scores"]],
                                "valid_precision": valid_results["precision"],
                                "valid_precision_scores": [valid_results["precision_scores"]],
                                "valid_recall": valid_results["recall"],
                                "valid_recall_scores": [valid_results["recall_scores"]]})

    if verbose_results:
        with pd.option_context('display.precision', 3):
            print(results)
        
    # remove any models that have been saved
    if os.path.exists(model_output):
        os.remove(model_output)
    
    return model, results