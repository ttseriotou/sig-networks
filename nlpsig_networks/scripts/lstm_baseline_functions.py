from __future__ import annotations

import nlpsig
from nlpsig_networks.pytorch_utils import _get_timestamp, SaveBestModel, set_seed
from nlpsig_networks.lstm_baseline import LSTMModel
from nlpsig_networks.scripts.implement_model import implement_model
from typing import Iterable
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os


def implement_lstm(
    num_epochs: int,
    x_data: torch.tensor | np.array,
    y_data: torch.tensor | np.array,
    hidden_dim: int,
    num_layers: int,
    bidirectional: bool,
    output_dim: int,
    dropout_rate: float,
    learning_rate: float,
    seed: int,
    loss: str,
    gamma: float = 0.0,
    batch_size: int = 64,
    data_split_seed: int = 0,
    split_ids: torch.Tensor | None = None,
    split_indices: tuple[Iterable[int], Iterable[int], Iterable[int]] | None = None,
    k_fold: bool = False,
    n_splits: int = 5,
    patience: int = 10,
    verbose_training: bool = True,
    verbose_results: bool = True,
    verbose_model: bool = False,
) -> tuple[LSTMModel, pd.DataFrame]:
    """
    Function which takes in input variables, x_data,
    and target output classification labels, y_data,
    and train and evaluates a LSTM (with dropout and ReLU activations). 
    
    If k_fold=True, it will evaluate the LSTM by performing k-fold validation
    with n_splits number of folds (i.e. train and test n_split number of LSTM),
    otherwise, it will evaluate by training and testing a single LSTM on one
    particular split of the data.

    Parameters
    ----------
    num_epochs : int
        Number of epochs
    x_data : torch.tensor | np.array
        Input variables
    y_data : torch.tensor | np.array
        Target classification labels
    hidden_dim : int
        Hidden dimensions in LSTM
    num_layers : int
        Number of recurrent layers.
    bidirectional : bool
        Whether or not a birectional LSTM is used,
        by default False (unidirectional LSTM is used in this case).
    output_dim : int
        Number of unique classification labels
    dropout_rate : float
        Droput rate to use in LSTM
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
    batch_size: int, optional
        Batch size, by default 64
    data_split_seed : int, optional
        The seed which is used when splitting, by default 0.
    split_ids : torch.Tensor | None, optional
        Groups to split by, default None.
    split_indices : tuple[Iterable[int], Iterable[int] | None, Iterable[int]] | None, optional
        Train, validation, test indices to use. If passed, will split the data
        according to these indices rather than splitting it within the method
        using the train_size and valid_size provided.
        First item in the tuple should be the indices for the training set,
        second item should be the indices for the validaton set (this could
        be None if no validation set is required), and third item should be
        indices for the test set.
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
    tuple[LSTMModel, pd.DataFrame]
        LSTMModel object (if k-fold, this is a randomly
        initialised model, otherwise it has been trained on the data splits
        that were generated within this function with data_split_seed),
        and dataframe of the evaluation metrics for the validation and
        test sets generated within this function.
    """
    # set seed
    set_seed(seed)
    
    # initialise LSTM
    lstm_model = LSTMModel(input_dim=x_data.shape[2],
                           hidden_dim=hidden_dim,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           output_dim=output_dim,
                           dropout_rate=dropout_rate)
    
    if verbose_model:
        print(lstm_model)
    
    # convert data to torch tensors
    if not isinstance(x_data, torch.Tensor):
        x_data = torch.tensor(x_data)
    if not isinstance(y_data, torch.Tensor):
        y_data = torch.tensor(y_data)
    x_data = x_data.float()
    
    return implement_model(model=lstm_model,
                           num_epochs=num_epochs,
                           x_data=x_data,
                           y_data=y_data,
                           learning_rate=learning_rate,
                           seed=seed,
                           loss=loss,
                           gamma=gamma,
                           batch_size=batch_size,
                           data_split_seed=data_split_seed,
                           split_ids=split_ids,
                           split_indices=split_indices,
                           k_fold=k_fold,
                           n_splits=n_splits,
                           patience=patience,
                           verbose_training=verbose_training,
                           verbose_results=verbose_results)

def obtain_path(df: pd.DataFrame,
                id_column: str,
                label_column: str,
                embeddings: np.array,
                k: int,
                path_indices : list | np.array | None = None) -> torch.tensor:
    # use nlpsig to construct the path as a numpy array
    # first define how we construct the path
    path_specifics = {"pad_by": "history",
                      "zero_padding": True,
                      "method": "k_last",
                      "k": k,
                      "features": None,
                      "embeddings": "full",
                      "include_current_embedding": True}
    
    # obtain path by using PrepareData class and .pad method
    paths = nlpsig.PrepareData(original_df=df,
                               id_column=id_column,
                               label_column=label_column,
                               embeddings=embeddings)
    path = paths.pad(**path_specifics)
    
    # slice the path in specified way
    if path_indices is not None:
        path = path[path_indices]

    # remove last two columns (which contains the id and the label)
    path = path[:,:,:-2].astype("float")

    return path


def lstm_hyperparameter_search(
    num_epochs: int,
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    embeddings: np.array,
    y_data: torch.tensor | np.array,
    output_dim: int,
    history_lengths: list[int],
    hidden_dim_sizes : list[int],
    num_layers: int,
    bidirectional: bool,
    dropout_rates: list[float],
    learning_rates: list[float],
    seeds : list[int],
    loss: str,
    gamma: float = 0.0,
    batch_size: int = 64,
    path_indices : list | np.array | None = None,
    data_split_seed: int = 0,
    split_ids: torch.Tensor | None = None,
    split_indices: tuple[Iterable[int], Iterable[int], Iterable[int]] | None = None,
    k_fold: bool = False,
    n_splits: int = 5,
    patience: int = 10,
    validation_metric: str = "f1",
    results_output: str | None = None,
    verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    """
    Performs hyperparameter search for different hidden dimensions,
    dropout rates, learning rates by training and evaluating
    a LSTM on various seeds and averaging performance over the seeds.
    
    If k_fold=True, will perform k-fold validation on each seed and
    average over the average performance over the folds, otherwise
    will average over the performance of the LSTMs trained using each
    seed.

    Parameters
    ----------
    num_epochs : int
        _description_
    x_data : torch.tensor | np.array
        _description_
    y_data : torch.tensor | np.array
        _description_
    output_dim : int
        Number of unique classification labels
    hidden_dim_sizes : list[int]
        _description_
    dropout_rates : list[float]
        _description_
    learning_rates : list[float]
        _description_
    seeds : list[int]
        _description_
    loss : str
        _description_
    gamma : float, optional
        _description_, by default 0.0
    batch_size : int, optional
        _description_, by default 64
    data_split_seed : int, optional
        _description_, by default 0
    split_ids : torch.Tensor | None, optional
        _description_, by default None  
    split_indices : tuple[Iterable[int], Iterable[int], Iterable[int]] | None, optional
        _description_, by default None 
    k_fold : bool, optional
        _description_, by default False
    n_splits : int, optional
        _description_, by default 5
    patience: int, optional
        _description_, by default 10
    validation_metric : str, optional
        _description_, by default "f1"
    results_output : str | None, optional
        _description_, by default None
    verbose : bool, optional
        _description_, by default True

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, float, dict]
        _description_
    """
    if validation_metric not in ["loss", "accuracy", "f1"]:
        raise ValueError("validation_metric must be either 'loss', 'accuracy' or 'f1'")
    
    # initialise SaveBestModel class
    model_output = f"best_lstm_model_{_get_timestamp()}.pkl"
    save_best_model = SaveBestModel(metric=validation_metric,
                                    output=model_output,
                                    verbose=verbose)

    # find model parameters that has the best validation
    results_df = pd.DataFrame()
    model_id = 0
    for k in tqdm(history_lengths):
        if verbose:
            print("\n" + "-" * 50)
            print(f"k: {k}")
            
        # obtain the lstm input by constructing a path from its history
        x_data = obtain_path(df=df,
                             id_column=id_column,
                             label_column=label_column,
                             embeddings=embeddings,
                             k=k,
                             path_indices=path_indices)

        for hidden_dim in tqdm(hidden_dim_sizes):
            for dropout in tqdm(dropout_rates):
                for lr in tqdm(learning_rates):
                    if verbose:
                        print("\n" + "!" * 50)
                        print(f"hidden_dim: {hidden_dim} | "
                            f"dropout: {dropout} | "
                            f"learning_rate: {lr}")
                    scores = []
                    verbose_model = verbose
                    for seed in seeds:
                        _, results = implement_lstm(num_epochs=num_epochs,
                                                    x_data=x_data,
                                                    y_data=y_data,
                                                    hidden_dim=hidden_dim,
                                                    num_layers=num_layers,
                                                    bidirectional=bidirectional,
                                                    output_dim=output_dim,
                                                    dropout_rate=dropout,
                                                    learning_rate=lr,
                                                    seed=seed,
                                                    loss=loss,
                                                    gamma=gamma,
                                                    batch_size=batch_size,
                                                    data_split_seed=data_split_seed,
                                                    split_ids=split_ids,
                                                    split_indices=split_indices,
                                                    k_fold=k_fold,
                                                    n_splits=n_splits,
                                                    patience = patience,
                                                    verbose_training=False,
                                                    verbose_results=verbose,
                                                    verbose_model=verbose_model)
                        
                        # save metric that we want to validate on
                        # taking the mean over the performance on the folds for the seed
                        # if k_fold=False, .mean() just returns the performance for the seed
                        scores.append(results[f"valid_{validation_metric}"].mean())
                        
                        # concatenate to results dataframe
                        results["k"] = k
                        results["num_layers"] = num_layers,
                        results["bidirectional"] = bidirectional
                        results["hidden_dim"] = hidden_dim
                        results["dropout_rate"] = dropout
                        results["learning_rate"] = lr
                        results["seed"] = seed
                        results["loss_function"] = loss
                        results["gamma"] = gamma
                        results["k_fold"] = k_fold
                        results["n_splits"] = n_splits if k_fold else None
                        results["batch_size"] = batch_size
                        results["model_id"] = model_id
                        results_df = pd.concat([results_df, results])
                        
                        # don't continue printing out the model
                        verbose_model = False
                    
                    model_id += 1
                    scores_mean = sum(scores)/len(scores)
                    
                    if verbose:
                        print(f"- average{' (kfold)' if k_fold else ''} "
                            f"(validation) metric score: {scores_mean}")
                        print(f"scores for the different seeds: {scores}")
                    # save best model according to averaged metric over the different seeds
                    save_best_model(current_valid_metric=scores_mean,
                                    extra_info={"k": k,
                                                "num_layers": num_layers,
                                                "bidirectional": bidirectional,
                                                "hidden_dim": hidden_dim,
                                                "dropout_rate": dropout,
                                                "learning_rate": lr})
                    
    checkpoint = torch.load(f=model_output)
    if verbose:
        print("*" * 50)
        print("The best model had the following parameters:")
        print(checkpoint["extra_info"])
    
    test_scores = []
    test_results_df = pd.DataFrame()
    for seed in seeds:
        _, test_results = implement_lstm(
            num_epochs=num_epochs,
            x_data=x_data,
            y_data=y_data,
            hidden_dim=checkpoint["extra_info"]["hidden_dim"],
            num_layers=checkpoint["extra_info"]["num_layers"],
            bidirectional=checkpoint["extra_info"]["bidirectional"],
            output_dim=output_dim,
            dropout_rate=checkpoint["extra_info"]["dropout_rate"],
            learning_rate=checkpoint["extra_info"]["learning_rate"],
            seed=seed,
            loss=loss,
            gamma=gamma,
            batch_size=batch_size,
            data_split_seed=data_split_seed,
            split_ids=split_ids,
            split_indices=split_indices,
            k_fold=k_fold,
            n_splits=n_splits,
            patience = patience,
            verbose_training=False,
            verbose_results=False
        )
        
        test_results["num_layers"] = checkpoint["extra_info"]["num_layers"]
        test_results["bidirectional"] = checkpoint["extra_info"]["bidirectional"]
        test_results["hidden_dim"] = checkpoint["extra_info"]["hidden_dim"]
        test_results["dropout_rate"] = checkpoint["extra_info"]["dropout_rate"]
        test_results["learning_rate"] = checkpoint["extra_info"]["learning_rate"]
        test_results["seed"] = seed
        test_results["loss_function"] = loss
        test_results["gamma"] = gamma
        test_results["k_fold"] = k_fold
        test_results["n_splits"] = n_splits if k_fold else None
        test_results["batch_size"] = batch_size
        test_results_df = pd.concat([test_results_df, test_results])
        
        # save metric that we want to validate on
        # taking the mean over the performance on the folds for the seed
        # if k_fold=False, .mean() just returns the performance for the seed
        test_scores.append(test_results[validation_metric].mean())
        
    test_scores_mean = sum(test_scores)/len(test_scores)
    if verbose:
        print(f"best validation score: {save_best_model.best_valid_metric}")
        print(f"- Best model: average (test) metric score: {test_scores_mean}")
        print(f"scores for the different seeds: {test_scores}")
        
    if results_output is not None:
        print("saving results dataframe to CSV for this "
              f"hyperparameter search in {results_output}")
        results_df.to_csv(results_output)
        best_results_output = results_output.replace(".csv", "_best_model.csv")
        print("saving the best model results dataframe to CSV for this "
              f"hyperparameter search in {best_results_output}")
        test_results_df.to_csv(best_results_output)
    
    # remove any models that have been saved
    if os.path.exists(model_output):
        os.remove(model_output)
    
    return results_df, test_results_df, save_best_model.best_valid_metric, checkpoint["extra_info"]
