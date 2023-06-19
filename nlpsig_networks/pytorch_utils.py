from __future__ import annotations
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm.auto import tqdm

from nlpsig.classification_utils import Folds, set_seed
from nlpsig_networks.focal_loss import ClassBalanced_FocalLoss, FocalLoss


class EarlyStopper:
    """
    Class to decide whether or not to stop training early by tracking
    the performance of the model on the validation set.
    """
    def __init__(self, metric: str, patience: int = 1, min_delta: float = 0.0):
        if metric not in ["loss", "accuracy", "f1"]:
            raise ValueError("metric must be either 'loss', 'accuracy' or 'f1'.")
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
                if self.counter > self.patience:
                    return True
        return False


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation metric is better than the previous best metric, then save the
    model state.
    
    If a metric on the training set is passed, we also track training metric progress,
    then save the model state if either the validation metric is strictly better
    than the previous best validation metric, or if the validation metric is 
    as good as the previous best validation metric and the training metric
    is better than the previous best training metric.
    """
    def __init__(self,
                 metric: str,
                 best_valid_metric: float | None = None,
                 best_train_metric: float | None = None,
                 output: str = "best_model.pkl",
                 verbose: bool = False):
        if metric not in ["loss", "accuracy", "f1"]:
            raise ValueError("metric must be either 'loss', 'accuracy' or 'f1'.")
        if best_valid_metric is None:
            if metric == "loss":
                best_valid_metric = float('inf')
            elif metric in ["accuracy", "f1"]:
                best_valid_metric = -float('inf')
        if best_train_metric is None:
            if metric == "loss":
                best_train_metric = float('inf')
            elif metric in ["accuracy", "f1"]:
                best_train_metric = -float('inf')
        self.metric = metric
        self.best_valid_metric = best_valid_metric
        self.best_train_metric = best_train_metric
        self.output = output
        self.verbose = verbose
        
    def __call__(self,
                 current_valid_metric: float,
                 model: nn.Module | None = None,
                 epoch: int | None = None,
                 current_train_metric: float | None = None,
                 extra_info: dict | str | None = None) -> None:
        if self.metric == "loss":
            # metric lower better
            if current_train_metric is not None:
                valid_condition = current_valid_metric <= self.best_valid_metric
                train_condition = current_train_metric < self.best_train_metric
            condition = current_valid_metric < self.best_valid_metric
        elif self.metric in ["accuracy", "f1"]:
            # metric higher better
            if current_train_metric is not None:
                valid_condition = current_valid_metric >= self.best_valid_metric
                train_condition = current_train_metric > self.best_train_metric
            condition = current_valid_metric > self.best_valid_metric
        if current_train_metric is not None:
            # we save if either the validation metric is strictly better
            # or the valid metric is at least as good as the best, but metric on train has improved
            condition = condition or (valid_condition and train_condition)
        if condition:
            self.best_valid_metric = current_valid_metric
            if self.verbose:
                print(f"New best validation metric: {self.best_valid_metric}")
                if epoch is not None:
                    print(f"Saving best model/info at epoch: {epoch+1} to {self.output}")
            torch.save(
                obj={
                    "model_state_dict": model.state_dict() if model is not None else None,
                    "epoch": epoch+1 if epoch is not None else epoch,
                    "extra_info": extra_info,
                },
                f=self.output,
            )


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
        Tuple with elements:
        - validation loss
        - validation accuracy
        - validation macro F1 score
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
        f1_scores = metrics.f1_score(labels, predicted, average=None)
        f1_v = sum(f1_scores)/len(f1_scores)
        if verbose:
            if epoch % verbose_epoch == 0:
                print(
                    f"[Validation] || Epoch: {epoch+1} || "
                    + f"Loss: {total_loss / len(valid_loader)} || "
                    + f"Accuracy: {accuracy} || "
                    + f"F1-score: {f1_v}"
                )

        return {"loss": total_loss / len(valid_loader),
                "accuracy": accuracy,
                "f1": f1_v,
                "f1_scores": f1_scores}


def training_pytorch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    scheduler: Optional[_LRScheduler] = None,
    valid_loader: Optional[DataLoader] = None,
    seed: Optional[int] = 42,
    save_best: bool = False,
    output: str = "best_model.pkl",
    early_stopping: bool = False,
    validation_metric: str = "loss",
    patience: Optional[int] = 10,
    verbose: bool = False,
    verbose_epoch: int = 100,
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
        Validation dataset as `torch.utils.data.dataloader.DataLoader` object
    criterion : torch.nn.Module
        Loss function which inherits from the `torch.nn.Module` class
    optimizer : torch.optim.optimizer.Optimizer
        PyTorch Optimizer
    num_epochs : int
        Number of epochs
    scheduler : Optional[_LRScheduler], optional
        Learning rate scheduler
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

    Returns
    -------
    torch.nn.Module
        Trained PyTorch model
    """
    if early_stopping and not isinstance(valid_loader, DataLoader):
        raise TypeError("if early stopping is required, need to pass in DataLoader object to `valid_loader`")
    if save_best and not isinstance(valid_loader, DataLoader):
        raise TypeError("if saving the best model is required, need to pass in DataLoader object to `valid_loader`")

    set_seed(seed)

    if save_best:
        # initialise SaveBestModel class
        save_best_model = SaveBestModel(metric=validation_metric,
                                        output=output,
                                        verbose=verbose)
    
    if early_stopping:
        # initialise EarlyStopper class
        early_stopper = EarlyStopper(metric=validation_metric,
                                     patience=patience,
                                     min_delta=0)

    # model train (& validation) per epoch
    if verbose:
        epochs_loop = tqdm(range(num_epochs))
    else:
        epochs_loop = range(num_epochs)
    for epoch in epochs_loop:
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
            if epoch % verbose_epoch == 0:
                print("-" * 50)
                print(
                    f"[Train] | Epoch: {epoch+1}/{num_epochs} || " + f"Loss: {loss.item()}"
                )

        if isinstance(valid_loader, DataLoader):
            # compute loss, accuracy and F1 on validation set
            validation_results = validation_pytorch(
                model=model,
                valid_loader=valid_loader,
                criterion=criterion,
                epoch=epoch,
                verbose=verbose,
                verbose_epoch=verbose_epoch,
            )
            
            # save metric that we want to use on validation set
            metric_v = validation_results[validation_metric]
                
            if save_best:
                # compute loss, accuracy and F1 on training set as well
                # this is to determine how well we're doing on the training set
                # allows us to choose between models that have the same validation
                validation_results = validation_pytorch(
                    model=model,
                    valid_loader=train_loader,
                    criterion=criterion,
                    epoch=epoch,
                    verbose=False,
                )
                # save metric that we want to validate on
                metric_train = validation_results[validation_metric]
                
                # save best model according to metric
                save_best_model(current_valid_metric=metric_v,
                                epoch=epoch,
                                model=model,
                                current_train_metric=metric_train)
                
            if early_stopping:
                # determine whether or not to stop early
                if early_stopper.early_stop(metric_v):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}!")
                    break
                
            if isinstance(scheduler, ReduceLROnPlateau):
                # use ReduceLROnPlateau to choose learning rate
                scheduler.step(validation_results["loss"])
        
        if (scheduler is not None) and (not isinstance(scheduler, ReduceLROnPlateau)):
            scheduler.step()

    if save_best:
        checkpoint = torch.load(f=output)
        model.load_state_dict(checkpoint["model_state_dict"])
        if verbose:
            print(f"Returning the best model which occurred at epoch {checkpoint['epoch']}")
        return model
    else:
        return model


def testing_pytorch(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    verbose: bool = True,
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
    verbose : bool, optional
        Whether or not to print progress, by default True

    Returns
    -------
    Tuple[Tuple[torch.tensor, torch.tensor], loss, accuracy, F1 score]
        Tuple with elements:
        - tuple of predicted labels and true labels
        - test loss
        - test accuracy
        - test macro F1 score
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
    
    # compute average loss
    avg_loss = total_loss / len(test_loader)
    # compute accuracy
    accuracy = ((predicted == labels).sum() / len(labels)).item()
    # compute F1
    f1_scores = metrics.f1_score(labels, predicted, average=None)
    f1 = sum(f1_scores)/len(f1_scores)
    
    if verbose:
        print(
            f"Accuracy on dataset of size {len(labels)}: "
            f"{100 * accuracy} %."
        )
        print(f"Average loss: {avg_loss}")
        print(f"- f1: {f1_scores}")
        print(f"- f1 (macro): {f1}")
    
    return {"predicted": predicted,
            "labels": labels,
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
            "f1_scores": f1_scores}


def KFold_pytorch(
    folds: Folds,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    seed: Optional[int] = 42,
    save_best: bool = False,
    early_stopping: bool = False,
    patience: Optional[int] = 10,
    verbose: bool = False,
    verbose_epoch: int = 100,
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
    verbose : bool, optional
        Whether or not to print progress, by default False
    verbose_epoch : int, optional
        How often to print progress during the epochs, by default 100

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
    loss = []
    accuracy = []
    f1 = []
    f1_scores = []
    valid_loss = []
    valid_accuracy = []
    valid_f1 = []
    valid_f1_scores = []
    fold_list = tqdm(range(folds.n_splits)) if verbose else range(folds.n_splits)
    for fold in fold_list:
        if verbose:
            print("\n" + "*" * 50)
            print(f"Fold: {fold+1} / {folds.n_splits}")
            print("*" * 50)

        # reload starting state of the model, optimizer and loss
        checkpoint = torch.load(f="starting_state.pkl")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        criterion = checkpoint["criterion"]
        if isinstance(criterion, FocalLoss):
            y_train = folds.get_splits(fold_index=fold)[1]
            criterion.set_alpha_from_y(y=y_train)
        elif isinstance(criterion, ClassBalanced_FocalLoss):
            y_train = folds.get_splits(fold_index=fold)[1]
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
            save_best=save_best,
            early_stopping=early_stopping,
            patience=patience,
            verbose=verbose,
            verbose_epoch=verbose_epoch,
        )

        # test model
        test_results = testing_pytorch(model=model,
                                       test_loader=test,
                                       criterion=criterion,
                                       verbose=verbose)

        # evaluate model
        loss.append(test_results["loss"])
        accuracy.append(test_results["accuracy"])
        f1.append(test_results["f1"])
        f1_scores.append(test_results["f1_scores"])
        
        if valid is not None:
            # test and evaluate on the validation set
            valid_results = testing_pytorch(model=model,
                                            test_loader=valid,
                                            criterion=criterion,
                                            verbose=verbose)
            valid_loss.append(valid_results["loss"])
            valid_accuracy.append(valid_results["accuracy"])
            valid_f1.append(valid_results["f1"])
            valid_f1_scores.append(valid_results["f1_scores"])
        else:
            valid_loss.append(None)
            valid_accuracy.append(None)
            valid_f1.append(None)
            valid_f1_scores.append(None)

    # remove starting state pickle file
    os.remove("starting_state.pkl")
    # if save_best=True, it will save models in best_model.pkl by default
    # so we remove
    if os.path.exists("best_model.pkl"):
        os.remove("best_model.pkl")
        
    return pd.DataFrame({"loss": loss,
                         "accuracy": accuracy, 
                         "f1": f1,
                         "f1_scores": f1_scores,
                         "valid_loss": valid_loss,
                         "valid_accuracy": valid_accuracy, 
                         "valid_f1": valid_f1,
                         "valid_f1_scores": valid_f1_scores})
