from __future__ import annotations
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm.auto import tqdm
import datetime

from nlpsig.classification_utils import Folds, set_seed
from nlpsig_networks.focal_loss import ClassBalanced_FocalLoss, FocalLoss


def _get_timestamp():
    timestamp = str(datetime.datetime.now())
    return timestamp.replace(" ", "-").replace(":", "-").replace(".", "-")


class EarlyStopper:
    """
    Class to decide whether or not to stop training early by tracking
    the performance of the model on the validation set.
    
    Class initialises a counter which will increase by 1 each time
    the validation metric gets worse. 
    """
    
    def __init__(self, metric: str, patience: int = 1, min_delta: float = 0.0):
        """
        Class to decide whether or not to stop training early by tracking
        the performance of the model on the validation set.
        
        Class initialises a counter which will increase by 1 each time
        the validation metric gets worse. 

        Parameters
        ----------
        metric : str
            Metric to use when deciding whether or not to stop training.
            Must be either "loss", "accuracy" or "f1" (for macro F1).
        patience : int, optional
            Patience to allow, i.e. the number of epochs allowed for
            the metric to get worse, by default 1.
        min_delta : float, optional
            Minimum amount of the metric has to get worse by
            in order to increase the count, by default 0.0.
        """
        if metric not in ["loss", "accuracy", "f1"]:
            raise ValueError("metric must be either 'loss', 'accuracy' or 'f1'.")
        
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation = np.inf
        self.max_validation = -np.inf

    def __call__(self, validation_metric: float) -> bool:
        """
        Method for determining whether or not to stop.
        
        If the validation metric gets worse (worse by an amount of self.min_delta),
        then will increase the counter. If the counter is larger than or
        equal to patience, then will return True, otherwise return False.
        
        If the validation metric is the best model so far, it will reset the counter to 0.

        Parameters
        ----------
        validation_metric : float
            The current metric obtained when evaluating
            the model on the validation set.

        Returns
        -------
        bool
            True to determine that we should stop training,
            False otherwise.
        """
        if self.metric == "loss":
            if validation_metric < self.min_validation:
                # a lower loss is better
                # we have a new best validation metric, so reset counter
                self.min_validation = validation_metric
                self.counter = 0
            elif validation_metric > (self.min_validation + self.min_delta):
                # new validation metric is worse than the best, increase counter
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        elif self.metric in ["accuracy", "f1"]:
            if validation_metric > self.max_validation:
                # a higher accuracy/F1 score is better
                # we have a new best validation metric, so reset counter
                self.max_validation = validation_metric
                self.counter = 0
            elif validation_metric < (self.max_validation - self.min_delta):
                # new validation metric is worse than the best, increase counter
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
                 output: str = f"save_best_model_{_get_timestamp()}.pkl",
                 verbose: bool = False):
        """
        Class to save the best model while training. If the current epoch's 
        validation metric is better than the previous best metric, then save the
        model state.
        
        If a metric on the training set is passed, we also track training metric progress,
        then save the model state if either the validation metric is strictly better
        than the previous best validation metric, or if the validation metric is 
        as good as the previous best validation metric and the training metric
        is better than the previous best training metric.

        Parameters
        ----------
        metric : str
            Metric to use when deciding whether or not to stop training.
            Must be either "loss", "accuracy" or "f1" (for macro F1).
        best_valid_metric : float | None, optional
            Current best metric on the validation set, by default None.
            If None, this will be set to infinity if metric="loss"
            (worse loss possible), otherwise will be set to -infinity
            if metric is either "accuracy" or "F1".
        best_train_metric : float | None, optional
            Current best metric on the train set, by default None.
            If None, this will be set to infinity if metric="loss"
            (worse loss possible), otherwise will be set to -infinity
            if metric is either "accuracy" or "F1".
            This can be used for making a decision between two models which
            have the same validation score.
        output : str, optional
            Where to store the best model, by default "save_best_model_{timestamp}.pkl".
            where timestamp is the time of initialising
        verbose : bool, optional
            Whether or not to print out progress, by default False.
        """
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
        """
        Method for determining whether or not to save current model.

        Parameters
        ----------
        current_valid_metric : float
            The current metric obtained when evaluating
            the model on the validation set.
        model : nn.Module | None, optional
            PyTorch model to save, by default None.
        epoch : int | None, optional
            Epoch number, by default None.
        current_train_metric : float | None, optional
            The current metric obtained when evaluating
            the model on the training set, by default None.
            This can be used for making a decision between two models which
            have the same validation score.
        extra_info : dict | str | None, optional
            Any extra information that you want to save to this model
            (if it does get saved), by default None.
        """
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
) -> dict[str, float | list[float]]:
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
    dict[str, float | list[float]]
    Dictionary with following items and keys:
        - "loss": average loss for the validation set
        - "accuracy": accuracy for the validation set
        - "f1": macro F1 score for the validation set
        - "f1_scores": F1 scores for each class in the validation set
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
        
        # compute accuracy
        accuracy = ((predicted == labels).sum() / len(labels)).item()
        
        # compute F1 scores
        f1_scores = metrics.f1_score(labels, predicted, average=None, zero_division=0.0)
        # compute macro F1 score
        f1 = sum(f1_scores)/len(f1_scores)
        
        # compute precision scores
        precision_scores = metrics.precision_score(labels, predicted, average=None, zero_division=0.0)
        # compute macro precision score
        precision = sum(precision_scores)/len(precision_scores)
        
        # compute recall scores
        recall_scores = metrics.recall_score(labels, predicted, average=None, zero_division=0.0)
        # compute macro recall score
        recall = sum(recall_scores)/len(recall_scores)
        
        if verbose:
            if epoch % verbose_epoch == 0:
                print(
                    f"[Validation] || Epoch: {epoch+1} || "
                    f"Loss: {total_loss / len(valid_loader)} || "
                    f"Accuracy: {accuracy} || "
                    f"F1-score: {f1} || "
                    f"Precision: {precision} ||"
                    f"Recall: {recall}"
                )

        return {"loss": total_loss / len(valid_loader),
                "accuracy": accuracy,
                "f1": f1,
                "f1_scores": f1_scores,
                "precision": precision,
                "precision_scores": precision_scores,
                "recall": recall,
                "recall_scores": recall_scores}


def training_pytorch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    scheduler: Optional[_LRScheduler] = None,
    valid_loader: Optional[DataLoader] = None,
    seed: Optional[int] = 42,
    return_best: bool = False,
    save_best: bool = False,
    output: str = f"best_model_{_get_timestamp()}.pkl",
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

    if save_best | return_best:
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
                
            if save_best | return_best:
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
                if early_stopper(metric_v):
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}!")
                    break
                
            if isinstance(scheduler, ReduceLROnPlateau):
                # use ReduceLROnPlateau to choose learning rate
                scheduler.step(validation_results["loss"])
        
        if (scheduler is not None) and (not isinstance(scheduler, ReduceLROnPlateau)):
            scheduler.step()

    if save_best | return_best:
        checkpoint = torch.load(f=output)
        model.load_state_dict(checkpoint["model_state_dict"])
        if save_best:
            if verbose:
                print(f"Returning the best model which occurred at epoch {checkpoint['epoch']}")
        if return_best:
            if not save_best:
                os.remove(output)
            return model
    
    return model


def testing_pytorch(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    verbose: bool = True,
) -> dict[str, torch.tensor | float | list[float]]:
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
    dict[str, torch.tensor | float | list[float]]
        Dictionary with following items and keys:
        - "predicted": torch.tensor containing the predicted labels
        - "labels": torch.tensor containing the true labels
        - "loss": average loss for the test set
        - "accuracy": accuracy for the test set
        - "f1": macro F1 score for the test set
        - "f1_scores": F1 scores for each class in the test set
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
    
    # compute F1 scores
    f1_scores = metrics.f1_score(labels, predicted, average=None, zero_division=0.0)
    # compute macro F1 score
    f1 = sum(f1_scores)/len(f1_scores)
    
    # compute precision scores
    precision_scores = metrics.precision_score(labels, predicted, average=None, zero_division=0.0)
    # compute macro precision score
    precision = sum(precision_scores)/len(precision_scores)
    
    # compute recall scores
    recall_scores = metrics.recall_score(labels, predicted, average=None, zero_division=0.0)
    # compute macro recall score
    recall = sum(recall_scores)/len(recall_scores)
    
    if verbose:
        print(
            f"Accuracy on dataset of size {len(labels)}: "
            f"{100 * accuracy} %."
        )
        print(f"Average loss: {avg_loss}")
        print(f"- f1: {f1_scores}")
        print(f"- f1 (macro): {f1}")
        print(f"- precision (macro): {precision}")
        print(f"- recall (macro): {recall}")
    
    return {"predicted": predicted,
            "labels": labels,
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
            "f1_scores": f1_scores,
            "precision": precision,
            "precision_scores": precision_scores,
            "recall": recall,
            "recall_scores": recall_scores}


def KFold_pytorch(
    folds: Folds,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    batch_size: int = 64,
    return_metric_for_each_fold: bool = False,
    seed: Optional[int] = 42,
    return_best: bool = False,
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
    batch_size : int, optional
        Batch size, by default 64
    return_metric_for_each_fold : bool, optional
        Whether or not to return the metrics for each fold individually,
        i.e. every row in the returned dataframe is the performance
        of the fitted model for each fold. If False, it will
        keep track of the predicted and true labels in the folds
        and return the overall metric for the dataset.
        If True, it will simply compute the metrics for each fold
        indvidually. One can then obtain a single metric by
        averaging over the performance over the different folds.
        By default False.
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
        Loss, Accuracy, F1 scores and macro F1 score for each fold
        (test and validation)
    """
    initial_starting_state_file = f"starting_state_{_get_timestamp()}.pkl"
    torch.save(
        obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "criterion": criterion,
        },
        f=initial_starting_state_file,
    )
    
    # create lists to record the test metrics for each fold
    loss = []
    accuracy = []
    f1 = []
    f1_scores = []
    precision = []
    precision_scores = []
    recall = []
    recall_scores = []
    
    # create lists to record the metrics evaluated on the 
    # validation sets for each fold
    valid_loss = []
    valid_accuracy = []
    valid_f1 = []
    valid_f1_scores = []
    valid_precision = []
    valid_precision_scores = []
    valid_recall = []
    valid_recall_scores = []
    
    # create empty torch tensors to record the predicted and labels
    labels = torch.empty((0))
    predicted = torch.empty((0))
    valid_labels = torch.empty((0))
    valid_predicted = torch.empty((0))
    
    # loop through folds to fit and evaluate
    fold_list = tqdm(range(folds.n_splits)) if verbose else range(folds.n_splits)
    for fold in fold_list:
        if verbose:
            print("\n" + "*" * 50)
            print(f"Fold: {fold+1} / {folds.n_splits}")
            print("*" * 50)

        # reload starting state of the model, optimizer and loss
        checkpoint = torch.load(f=initial_starting_state_file)
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
        data_loader_args = {"batch_size": batch_size, "shuffle": True}
        train, valid, test = folds.get_splits(fold_index=fold, as_DataLoader=True, data_loader_args=data_loader_args)

        # train pytorch model
        model = training_pytorch(
            model=model,
            train_loader=train,
            valid_loader=valid,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            seed=seed,
            return_best=return_best,
            save_best=False,
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

        # store the true labels and predicted labels for this fold
        labels = torch.cat([labels, test_results["labels"]])
        predicted = torch.cat([predicted, test_results["predicted"]])
        
        # evaluate model
        loss.append(test_results["loss"])
        accuracy.append(test_results["accuracy"])
        f1.append(test_results["f1"])
        f1_scores.append(test_results["f1_scores"])
        precision.append(test_results["precision"])
        precision_scores.append(test_results["precision_scores"])
        recall.append(test_results["recall"])
        recall_scores.append(test_results["recall_scores"])
        
        if valid is not None:
            # test and evaluate on the validation set
            valid_results = testing_pytorch(model=model,
                                            test_loader=valid,
                                            criterion=criterion,
                                            verbose=verbose)
            
            # store the true labels and predicted labels for this fold
            valid_labels = torch.cat([valid_labels, valid_results["labels"]])
            valid_predicted = torch.cat([valid_predicted, valid_results["predicted"]])
            
            # store the metrics for the validation set
            valid_loss.append(valid_results["loss"])
            valid_accuracy.append(valid_results["accuracy"])
            valid_f1.append(valid_results["f1"])
            valid_f1_scores.append(valid_results["f1_scores"])
            valid_precision.append(test_results["precision"])
            valid_precision_scores.append(test_results["precision_scores"])
            valid_recall.append(test_results["recall"])
            valid_recall_scores.append(test_results["recall_scores"])
        else:
            valid_loss.append(None)
            valid_accuracy.append(None)
            valid_f1.append(None)
            valid_f1_scores.append(None)
            valid_precision.append(None)
            valid_precision_scores.append(None)
            valid_recall.append(None)
            valid_recall_scores.append(None)

    # remove starting state pickle file
    os.remove(initial_starting_state_file)
    
    if return_metric_for_each_fold:
        # return how well the model performed on each individual fold
        return pd.DataFrame({"loss": loss,
                             "accuracy": accuracy, 
                             "f1": f1,
                             "f1_scores": f1_scores,
                             "precision": precision,
                             "precision_scores": precision_scores,
                             "recall": recall,
                             "recall_scores": recall_scores,
                             "valid_loss": valid_loss,
                             "valid_accuracy": valid_accuracy, 
                             "valid_f1": valid_f1,
                             "valid_f1_scores": valid_f1_scores,
                             "valid_precision": valid_precision,
                             "valid_precision_scores": valid_precision_scores,
                             "valid_recall": valid_recall,
                             "valid_recall_scores": valid_recall_scores})
    else:
        # compute how well the model performed on the test sets together
        # compute accuracy
        accuracy = ((predicted == labels).sum() / len(labels)).item()
        # compute F1
        f1_scores = metrics.f1_score(labels, predicted, average=None, zero_division=0.0)
        f1 = sum(f1_scores)/len(f1_scores)
        
        # compute precision scores
        precision_scores = metrics.precision_score(labels, predicted, average=None, zero_division=0.0)
        # compute macro precision score
        precision = sum(precision_scores)/len(precision_scores)
        
        # compute recall scores
        recall_scores = metrics.recall_score(labels, predicted, average=None, zero_division=0.0)
        # compute macro recall score
        recall = sum(recall_scores)/len(recall_scores)
        
        if valid is not None:
            # compute how well the model performed on the
            # validation sets in the folds
            valid_accuracy = ((valid_predicted == valid_labels).sum() / len(valid_labels)).item()
            
            # compute F1
            valid_f1_scores = metrics.f1_score(valid_labels, valid_predicted, average=None, zero_division=0.0)
            valid_f1 = sum(valid_f1_scores)/len(valid_f1_scores)
            
            # compute precision scores
            valid_precision_scores = metrics.precision_score(valid_labels, valid_predicted, average=None, zero_division=0.0)
            # compute macro precision score
            valid_precision = sum(valid_precision_scores)/len(valid_precision_scores)
            
            # compute recall scores
            valid_recall_scores = metrics.recall_score(valid_labels, valid_predicted, average=None, zero_division=0.0)
            # compute macro recall score
            valid_recall = sum(valid_recall_scores)/len(valid_recall_scores)
        else:
            valid_accuracy = None
            valid_f1 = None
            valid_f1_scores = None
            valid_precision = None
            valid_precision_scores = None
            valid_recall = None
            valid_recall_scores = None
        
        return pd.DataFrame({"loss": None,
                             "accuracy": accuracy, 
                             "f1": f1,
                             "f1_scores": [f1_scores],
                             "precision": precision,
                             "precision_scores": [precision_scores],
                             "recall": recall,
                             "recall_scores": [recall_scores],
                             "valid_loss": None,
                             "valid_accuracy": valid_accuracy, 
                             "valid_f1": valid_f1,
                             "valid_f1_scores": [valid_f1_scores],
                             "valid_precision": valid_precision,
                             "valid_precision_scores": [valid_precision_scores],
                             "valid_recall": valid_recall,
                             "valid_recall_scores": [valid_recall_scores]})