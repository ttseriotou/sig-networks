import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from nlpsig.classification_utils import Folds, set_seed
from nlpsig_networks.focal_loss import ClassBalanced_FocalLoss, FocalLoss


class EarlyStopper:
    def __init__(self, metric: str, patience: int = 1, min_delta: float = 0.0):
        if metric not in ["loss", "accuracy", "f1"]:
            raise ValueError("metric must be either 'loss', 'accuracy' or 'f1'. ")
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation = np.inf
        self.max_validation = -np.inf

    def early_stop(self, validation_metric: float) -> bool:
        if self.metric == "loss":
            if validation_metric < self.min_validation:
                self.min_validation = validation_metric
                self.counter = 0
            elif validation_metric > (self.min_validation + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        elif self.metric in ["accuracy", "f1"]:
            if validation_metric > self.max_validation:
                self.max_validation = validation_metric
                self.counter = 0
            elif validation_metric < (self.max_validation - self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False


def validation_pytorch(
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    verbose: bool = False,
    verbose_epoch: int = 100,
) -> Tuple[float, float, float]:
    """
    Evaluates the PyTorch model to a validation set and
    returns the total loss, accuracy and F1 score

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model which inherits from the `torch.nn.Module` class
    valid_loader : DataLoader
        Validation dataset as `torch.utils.data.dataloader.DataLoader` object
    criterion : torch.nn.Module
        Loss function which inherits from the `torch.nn.Module` class
    epoch : int
        Epoch number
    verbose : bool, optional
        Whether or not to print progress, by default False
    verbose_epoch : int, optional
        How often to print progress during the epochs, by default 100

    Returns
    -------
    Tuple[float, float, float]
        Current average loss, accuracy and F1 score
    """
    # sets the model to evaluation mode
    model.eval()
    total_loss = 0
    labels = torch.empty((0))
    predicted = torch.empty((0))
    with torch.no_grad():
        for emb_v, labels_v in valid_loader:
            # make prediction
            outputs = model(emb_v)
            _, predicted_v = torch.max(outputs.data, 1)
            # compute loss
            total_loss += criterion(outputs, labels_v).item()
            # save predictions and labels
            labels = torch.cat([labels, labels_v])
            predicted = torch.cat([predicted, predicted_v])
        # compute accuracy and f1 score
        accuracy = ((predicted == labels).sum() / len(labels)).item()
        f1_v = metrics.f1_score(labels, predicted, average="macro")
        if verbose:
            if epoch % verbose_epoch == 0:
                print(
                    f"Validation || Epoch: {epoch+1} || "
                    + f"Loss: {total_loss / len(valid_loader)} || "
                    + f"Accuracy: {accuracy} || "
                    + f"F1-score: {f1_v}"
                )

        return total_loss / len(valid_loader), accuracy, f1_v


def training_pytorch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    valid_loader: Optional[DataLoader] = None,
    seed: Optional[int] = 42,
    early_stopping: bool = False,
    early_stopping_metric: str = "loss",
    patience: Optional[int] = 10,
    verbose: bool = False,
    verbose_epoch: int = 100,
    verbose_item: int = 1000,
) -> nn.Module:
    """
    Trains the PyTorch model using some training dataset and
    uses a validation dataset to determine if early stopping is used

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model which inherits from the `torch.nn.Module` class
    train_loader : torch.utils.data.dataloader.DataLoader
        Training dataset as `torch.utils.data.dataloader.DataLoader` object
    valid_loader : torch.utils.data.dataloader.DataLoader
        Validation dataset as `torch.utils.data.dataloader.DataLoader` objectorc
    criterion : torch.nn.Module
        Loss function which inherits from the `torch.nn.Module` class
    optimizer : torch.optim.optimizer.Optimizer
        PyTorch Optimizer
    num_epochs : int
        Number of epochs
    seed : Optional[int], optional
        Seed number, by default 42
    early_stopping: bool, optional
        Whether or not early stopping will be done, in which case
        you must consider the `patience` argument
    patience : Optional[int], optional
        Patience parameter for early stopping rule, by default 10
    verbose : bool, optional
        Whether or not to print progress, by default False
    verbose_epoch : int, optional
        How often to print progress during the epochs, by default 100
    verbose_item : int, optional
        How often to print progress when iterating over items
        in training set, by default 1000

    Returns
    -------
    torch.nn.Module
        Trained PyTorch model
    """
    if early_stopping and type(valid_loader) != DataLoader:
        raise TypeError("if early stopping is required, need to pass in DataLoader object to `valid_loader`")

    set_seed(seed)

    # early stopping parameters
    if early_stopping:
        early_stopper = EarlyStopper(metric=early_stopping_metric,
                                     patience=patience,
                                     min_delta=0)

    # model train & validation per epoch
    for epoch in tqdm(range(num_epochs)):
        for i, (emb, labels) in enumerate(train_loader):
            # sets the model to training mode
            model.train()
            # perform training by performing forward and backward passes
            optimizer.zero_grad()
            outputs = model(emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # show training progress
            if verbose:
                if (epoch % verbose_epoch == 0) and (i % verbose_item == 0):
                    print(
                        f"Epoch: {epoch+1}/{num_epochs} || "
                        + f"Item: {i}/{len(train_loader)} || "
                        + f"Loss: {loss.item()}"
                    )

        # show training progress
        if verbose:
            if epoch % verbose_epoch == 0:
                print("-" * 50)
                print(
                    f"##### Epoch: {epoch+1}/{num_epochs} || " + f"Loss: {loss.item()}"
                )
                print("-" * 50)

        # determine whether or not to stop early using validation set
        if early_stopping:
            loss_v, acc_v, f1_v = validation_pytorch(
                model=model,
                valid_loader=valid_loader,
                criterion=criterion,
                epoch=epoch,
                verbose=verbose,
                verbose_epoch=verbose_epoch,
            )
            if early_stopping_metric == "loss":
                validation_metric = loss_v
            elif early_stopping_metric == "accuracy":
                validation_metric = acc_v
            elif early_stopping_metric == "f1":
                validation_metric = f1_v
            if early_stopper.early_stop(validation_metric):
                print(f"Early stopping at epoch {epoch+1}!")
                break

    return model


def testing_pytorch(
    model: nn.Module, test_loader: DataLoader, criterion: nn.Module,
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Evaluates the PyTorch model to a validation set and
    returns the predicted labels and their corresponding true labels

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model which inherits from the `torch.nn.Module` class
    test_loader : DataLoader
        Testing dataset as `torch.utils.data.dataloader.DataLoader` object
    criterion : torch.nn.Module
        Loss function which inherits from the `torch.nn.Module` class

    Returns
    -------
    Tuple[torch.tensor, torch.tensor]
        Predicted labels and true labels
    """
    # sets the model to evaluation mode
    model.eval()
    total_loss = 0
    labels = torch.empty((0))
    predicted = torch.empty((0))
    with torch.no_grad():
        # Iterate through test dataset
        for emb_t, labels_t in test_loader:
            # make prediction
            outputs_t = model(emb_t)
            _, predicted_t = torch.max(outputs_t.data, 1)
            # compute loss
            total_loss += criterion(outputs_t, labels_t).item()
            # save predictions and labels
            labels = torch.cat([labels, labels_t])
            predicted = torch.cat([predicted, predicted_t])
            

    print(
        f"Accuracy on dataset of size {len(labels)}: "
        f"{100 * sum(labels==predicted) / len(labels)} %."
    )
    print(f"Average loss: {total_loss / len(test_loader)}")
    return predicted, labels


def KFold_pytorch(
    folds: Folds,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    seed: Optional[int] = 42,
    early_stopping: bool = False,
    patience: Optional[int] = 10,
    verbose_args: dict = {
        "verbose": True,
        "verbose_epoch": 100,
        "verbose_item": 10000,
    },
) -> pd.DataFrame:
    """
    Performs KFold evaluation for a PyTorch model

    Parameters
    ----------
    folds : Folds
        Object which stores and obtains the folds
    model : torch.nn.Module
        PyTorch model which inherits from the `torch.nn.Module` class
    criterion : torch.nn.Module
        Loss function which inherits from the `torch.nn.Module` class
    optimizer : torch.optim.optimizer.Optimizer
        PyTorch Optimizer
    num_epochs : int
        Number of epochs
    seed : Optional[int], optional
        Seed number, by default 42
    early_stopping: bool, optional
        Whether or not early stopping will be done, in which case
        you must consider the `patience` argument
    patience : Optional[int], optional
        Patience parameter for early stopping rule, by default 10
    verbose_args : _type_, optional
        Arguments for how to print progress, by default
        {"verbose": True,
         "verbose_epoch": 100,
         "verbose_item": 10000}

    Returns
    -------
    pd.DataFrame
        Accuracy and F1 scores for each fold
    """
    torch.save(
        obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "criterion": criterion,
        },
        f="starting_state.pkl",
    )
    accuracy = []
    f1_score = []
    for fold in tqdm(range(folds.n_splits)):
        print("\n" + "*" * 50)
        print(f"Fold: {fold+1} / {folds.n_splits}")
        print("*" * 50)

        # reload starting state of the model, optimizer and loss
        checkpoint = torch.load(f="starting_state.pkl")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        criterion = checkpoint["criterion"]
        if isinstance(criterion, FocalLoss):
            y_train = folds.get_splits(fold_index=fold)[5]
            criterion.set_alpha_from_y(y=y_train)
        elif isinstance(criterion, ClassBalanced_FocalLoss):
            y_train = folds.get_splits(fold_index=fold)[5]
            criterion.set_samples_per_cls_from_y(y=y_train)

        # obtain test, valid and test dataloaders
        train, valid, test = folds.get_splits(fold_index=fold, as_DataLoader=True)

        # train pytorch model
        model = training_pytorch(
            model=model,
            train_loader=train,
            valid_loader=valid,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            seed=seed,
            early_stopping=early_stopping,
            patience=patience,
            **verbose_args,
        )

        # test model
        predicted, labels = testing_pytorch(model=model, test_loader=test)

        # evaluate model
        accuracy.append(((predicted == labels).sum() / len(labels)).item())
        f1_score.append(metrics.f1_score(labels, predicted, average="macro"))

    # remove starting state pickle file
    os.remove("starting_state.pkl")
    return pd.DataFrame({"accuracy": accuracy, "f1_score": f1_score})
