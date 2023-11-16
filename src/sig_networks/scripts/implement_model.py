from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nlpsig.classification_utils import DataSplits, Folds

from sig_networks.focal_loss import FocalLoss
from sig_networks.pytorch_utils import (
    KFold_pytorch,
    _get_timestamp,
    testing_pytorch,
    training_pytorch,
)


def implement_model(
    model: nn.Module,
    num_epochs: int,
    x_data: np.array | torch.Tensor | dict[str, np.array | torch.Tensor],
    y_data: torch.tensor | np.array,
    learning_rate: float,
    seed: int,
    loss: str,
    gamma: float = 0.0,
    device: str | None = None,
    batch_size: int = 64,
    data_split_seed: int = 0,
    split_ids: torch.Tensor | None = None,
    split_indices: tuple[Iterable[int] | None] | None = None,
    k_fold: bool = False,
    n_splits: int = 5,
    patience: int | None = 10,
    verbose_training: bool = False,
    verbose_results: bool = False,
) -> tuple[nn.Module, pd.DataFrame]:
    """
    Helper function to implement a model using PyTorch.

    Can be used for both k-fold cross-validation and train-validation-test splits.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model which inherits from the `torch.nn.Module` class
    num_epochs : int
        Number of epochs
    x_data : np.array | torch.Tensor | dict[str, np.array | torch.Tensor]
        Input variables. This can be standard numpy arrays or torch tensors, and
        can be a dictionary of arrays/tensors if the model expects multiple inputs
    y_data : torch.tensor | np.array
        Target classification labels
    learning_rate : float
        Learning rate to use
    seed : int
        Seed to use throughout (besides for splitting the data - see data_split_seed)
    loss : str
        Loss to use, options are "focal" for focal loss, and "cross_entropy" for
        cross-entropy loss
    gamma : float, optional
        Gamma to use for focal loss, by default 0.0.
        Ignored if loss="cross_entropy"
    device : str | None, optional
        Device to use for training and evaluation, by default None
    batch_size: int, optional
        Batch size, by default 64
    data_split_seed : int, optional
        The seed which is used when splitting, by default 0
    split_ids : torch.Tensor | None, optional
        Groups to split by, default None
    split_indices : tuple[Iterable[int] | None] | None, optional
        Train, validation, test indices to use. If passed, will split the data
        according to these indices rather than splitting it within the method
        using the train_size and valid_size provided.
        First item in the tuple should be the indices for the training set,
        second item should be the indices for the validaton set (this could
        be None if no validation set is required), and third item should be
        indices for the test set
    k_fold : bool, optional
        Whether or not to use k-fold validation, by default False
    n_splits : int, optional
        Number of splits to use in k-fold validation, by default 5.
        Ignored if k_fold=False
    patience : int, optional
        Patience of training, by default 10.
    verbose_training : bool, optional
        Whether or not to print out training progress, by default True
    verbose_results : bool, optional
        Whether or not to print out results on validation and test, by default True


    Returns
    -------
    tuple[torch.nn.Module, pd.DataFrame]
        The trained model (if k-fold, this is a randomly
        initialised model, otherwise it has been trained on the data splits
        that were generated within this function with data_split_seed),
        and dataframe of the evaluation metrics for the validation and
        test sets generated within this function.
    """
    # set some variables for training
    return_best = True
    early_stopping = patience is not None
    model_output = f"best_model_{_get_timestamp()}.pkl"
    validation_metric = "f1"
    weight_decay_adam = 0.0001

    if device is not None:
        # set model to device is passed
        model.to(device)

        # convert data to tensors if passed as numpy arrays
        # deal with case if x_data is a dictionary
        if isinstance(x_data, dict):
            # iterate through the values and check they are of the correct type
            for key, value in x_data.items():
                if isinstance(value, np.ndarray):
                    x_data[key] = torch.from_numpy(value)
                # set data to device
                x_data[key] = x_data[key].to(device)
        # deal with case if x_data is a numpy array
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        if isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)

        # set data to device
        if isinstance(x_data, dict):
            # iterate through the values and send to device
            for key, value in x_data.items():
                x_data[key] = x_data[key].to(device)
        else:
            x_data = x_data.to(device)
        y_data = y_data.to(device)

    if k_fold:
        # perform KFold evaluation and return the performance
        # on validation and test sets
        # first split dataset
        folds = Folds(
            x_data=x_data,
            y_data=y_data,
            groups=split_ids,
            n_splits=n_splits,
            indices=split_indices,
            shuffle=True,
            random_state=data_split_seed,
        )

        # define loss
        if loss == "focal":
            criterion = FocalLoss(gamma=gamma)
        elif loss == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("criterion must be either 'focal' or 'cross_entropy'")

        # define optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay_adam
        )

        # perform k-fold evaluation which returns a dataframe with columns for the
        # loss, accuracy, f1 (macro) and individual f1-scores for each fold
        # (for both validation and test set)
        results = KFold_pytorch(
            folds=folds,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            batch_size=batch_size,
            seed=seed,
            return_best=return_best,
            early_stopping=early_stopping,
            patience=patience,
            device=device,
            verbose=verbose_training,
        )
    else:
        # split dataset
        data_loader_args = {"batch_size": batch_size, "shuffle": True}

        split_data = DataSplits(
            x_data=x_data,
            y_data=y_data,
            groups=split_ids,
            train_size=0.8,
            valid_size=0.2,
            indices=split_indices,
            shuffle=True,
            random_state=data_split_seed,
        )
        train, valid, test = split_data.get_splits(
            as_DataLoader=True, data_loader_args=data_loader_args
        )

        # define loss
        if loss == "focal":
            criterion = FocalLoss(gamma=gamma)
            y_train = split_data.get_splits(as_DataLoader=False)[1]
            criterion.set_alpha_from_y(y=y_train)
        elif loss == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("criterion must be either 'focal' or 'cross_entropy'")

        # define optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay_adam
        )

        # train model
        model = training_pytorch(
            model=model,
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
            device=device,
            verbose=verbose_training,
        )

        # evaluate on validation
        valid_results = testing_pytorch(
            model=model,
            test_loader=valid,
            criterion=criterion,
            device=device,
            verbose=False,
        )

        # evaluate on test
        test_results = testing_pytorch(
            model=model,
            test_loader=test,
            criterion=criterion,
            device=device,
            verbose=False,
        )

        results = pd.DataFrame(
            {
                "loss": test_results["loss"],
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
                "valid_recall_scores": [valid_results["recall_scores"]],
            }
        )

    if verbose_results:
        with pd.option_context("display.precision", 3):
            print(results)

    # remove any models that have been saved
    if os.path.exists(model_output):
        os.remove(model_output)

    return model, results
