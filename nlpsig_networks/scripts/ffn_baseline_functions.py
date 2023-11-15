from __future__ import annotations

import os
from typing import Iterable

import nlpsig
import numpy as np
import pandas as pd
import signatory
import torch
from tqdm.auto import tqdm

from nlpsig_networks.ffn_baseline import FeedforwardNeuralNetModel
from nlpsig_networks.pytorch_utils import SaveBestModel, _get_timestamp, set_seed
from nlpsig_networks.scripts.implement_model import implement_model


def implement_ffn(
    num_epochs: int,
    x_data: torch.tensor | np.array,
    y_data: torch.tensor | np.array,
    hidden_dim: list[int] | int,
    output_dim: int,
    dropout_rate: float,
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
    patience: int = 10,
    verbose_training: bool = False,
    verbose_results: bool = False,
    verbose_model: bool = False,
) -> tuple[FeedforwardNeuralNetModel, pd.DataFrame]:
    """
    Function which takes in input variables, x_data,
    and target output classification labels, y_data,
    and train and evaluates a FFN (with dropout and ReLU activations).

    If k_fold=True, it will evaluate the FFN by performing k-fold validation
    with n_splits number of folds (i.e. train and test n_split number of FFN),
    otherwise, it will evaluate by training and testing a single FFN on one
    particular split of the data.

    Parameters
    ----------
    num_epochs : int
        Number of epochs
    x_data : torch.tensor | np.array
        Input variables
    y_data : torch.tensor | np.array
        Target classification labels
    hidden_dim : list[int] | int
        Hidden dimensions in FFN, can be int if a single hidden layer,
        or can be a list of ints for multiple hidden layers
    output_dim : int
        Number of unique classification labels
    dropout_rate : float
        Droput rate to use in FFN
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
        Whether or not to print out training progress, by default False
    verbose_results : bool, optional
        Whether or not to print out results on validation and test, by default False
    verbose_model : bool, optional
        Whether or not to print out the model, by default False

    Returns
    -------
    tuple[FeedforwardNeuralNetModel, pd.DataFrame]
        FeedforwardNeuralNetModel object (if k-fold, this is a randomly
        initialised model, otherwise it has been trained on the data splits
        that were generated within this function with data_split_seed),
        and dataframe of the evaluation metrics for the validation and
        test sets generated within this function.
    """
    # set seed
    set_seed(seed)

    # initialise FFN
    ffn_model = FeedforwardNeuralNetModel(
        input_dim=x_data.shape[1],
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    )

    if verbose_model:
        print(ffn_model)

    # convert data to torch tensors
    if not isinstance(x_data, torch.Tensor):
        x_data = torch.tensor(x_data)
    if not isinstance(y_data, torch.Tensor):
        y_data = torch.tensor(y_data)
    x_data = x_data.float()

    return implement_model(
        model=ffn_model,
        num_epochs=num_epochs,
        x_data=x_data,
        y_data=y_data,
        learning_rate=learning_rate,
        seed=seed,
        loss=loss,
        gamma=gamma,
        device=device,
        batch_size=batch_size,
        data_split_seed=data_split_seed,
        split_ids=split_ids,
        split_indices=split_indices,
        k_fold=k_fold,
        n_splits=n_splits,
        patience=patience,
        verbose_training=verbose_training,
        verbose_results=verbose_results,
    )


def ffn_hyperparameter_search(
    num_epochs: int,
    x_data: torch.tensor | np.array,
    y_data: torch.tensor | np.array,
    output_dim: int,
    hidden_dim_sizes: list[list[int]] | list[int],
    dropout_rates: list[float],
    learning_rates: list[float],
    seeds: list[int],
    loss: str,
    gamma: float = 0.0,
    device: str | None = None,
    batch_size: int = 64,
    data_split_seed: int = 0,
    split_ids: torch.Tensor | None = None,
    split_indices: tuple[Iterable[int] | None] | None = None,
    k_fold: bool = False,
    n_splits: int = 5,
    patience: int = 10,
    validation_metric: str = "f1",
    results_output: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    """
    Performs hyperparameter search for the baseline FFN model
    for different hidden dimensions, dropout rates, learning rates
    by training and evaluating a FFN on various seeds and
    averaging performance over the seeds.

    We select the best model based on the average performance on
    the validation set. We then evaluate the best model on the test set.

    If k_fold=True, will perform k-fold validation on each seed and
    average over the average performance over the folds, otherwise
    will average over the performance of the FFNs trained using each
    seed.

    Parameters
    ----------
    num_epochs : int
        Number of epochs
    x_data : torch.tensor | np.array
        Input variables
    y_data : torch.tensor | np.array
        Target classification labels
    output_dim : int
        Number of unique classification labels
    hidden_dim_sizes : list[list[int]] | list[int]
        Hidden dimensions in FFN to try out. Each element in the list
        should be a list of ints if multiple hidden layers are required,
        or an int if a single hidden layer is required
    dropout_rates : list[float]
        Dropout rates to try out. Each element in the list
        should be a float
    learning_rates : list[float]
        Learning rates to try out. Each element in the list
        should be a float
    seeds : list[int]
        Seeds to use throughout to average over the performance
        (besides for splitting the data - see data_split_seed)
    loss : str
        Loss to use, options are "focal" for focal loss, and
        "cross_entropy" for cross-entropy loss.
    gamma : float, optional
        Value of gamma in focal loss, by default 0.0.
        Ignored if loss="cross_entropy".
    batch_size: int, optional
        Batch size to use in training, by default 64.
    data_split_seed : int, optional
        The seed which is used when splitting, by default 0
    split_ids : torch.Tensor | None, optional
        Groups to split by, default None.
    split_indices : tuple[Iterable[int] | None] | None, optional
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
    validation_metric : str, optional
        Metric to use to use for determining the best model, by default "f1"
    results_output : str | None, optional
        Path for where to save the results dataframe, by default None
    verbose : bool, optional
        Whether or not to print out progress, by default True

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, float, dict]
        A tuple containing the full results dataframe which includes the
        performance of each of the models fitted during the hyperparameter
        search for each of the seeds, the results dataframe for the best
        performing model (based on the average validation metric performance),
        the average validation metric performance of the best model, and
        the hyperparameters which gave the best model.
    """
    if validation_metric not in ["loss", "accuracy", "f1"]:
        raise ValueError("validation_metric must be either 'loss', 'accuracy' or 'f1'")

    # initialise SaveBestModel class
    model_output = f"best_ffn_model_{_get_timestamp()}.pkl"
    save_best_model = SaveBestModel(
        metric=validation_metric, output=model_output, verbose=verbose
    )

    # find model parameters that has the best validation
    results_df = pd.DataFrame()
    model_id = 0
    for hidden_dim in tqdm(hidden_dim_sizes):
        for dropout in tqdm(dropout_rates):
            for lr in tqdm(learning_rates):
                if verbose:
                    print("\n" + "!" * 50)
                    print(
                        f"hidden_dim: {hidden_dim} | "
                        f"dropout: {dropout} | "
                        f"learning_rate: {lr}"
                    )
                scores = []
                verbose_model = verbose
                for seed in seeds:
                    _, results = implement_ffn(
                        num_epochs=num_epochs,
                        x_data=x_data,
                        y_data=y_data,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        dropout_rate=dropout,
                        learning_rate=lr,
                        seed=seed,
                        loss=loss,
                        gamma=gamma,
                        device=device,
                        batch_size=batch_size,
                        data_split_seed=data_split_seed,
                        split_ids=split_ids,
                        split_indices=split_indices,
                        k_fold=k_fold,
                        n_splits=n_splits,
                        patience=patience,
                        verbose_training=False,
                        verbose_results=verbose,
                        verbose_model=verbose_model,
                    )

                    # save metric that we want to validate on
                    # take mean of performance on the folds
                    # if k_fold=False, return performance for seed
                    scores.append(results[f"valid_{validation_metric}"].mean())

                    # concatenate to results dataframe
                    results["hidden_dim"] = [
                        tuple(hidden_dim) for _ in range(len(results.index))
                    ]
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

                scores_mean = sum(scores) / len(scores)

                if verbose:
                    print(
                        f"- average{' (kfold)' if k_fold else ''} "
                        f"(validation) metric score: {scores_mean}"
                    )
                    print(f"scores for the different seeds: {scores}")

                # save best model according to averaged metric over the different seeds
                # if the score is better than the previous best,
                # we save the parameters to model_output
                save_best_model(
                    current_valid_metric=scores_mean,
                    extra_info={
                        "hidden_dim": hidden_dim,
                        "dropout_rate": dropout,
                        "learning_rate": lr,
                    },
                )

    checkpoint = torch.load(f=model_output)
    if verbose:
        print("*" * 50)
        print("The best model had the following parameters:")
        print(checkpoint["extra_info"])

    test_scores = []
    test_results_df = pd.DataFrame()
    for seed in seeds:
        _, test_results = implement_ffn(
            num_epochs=num_epochs,
            x_data=x_data,
            y_data=y_data,
            hidden_dim=checkpoint["extra_info"]["hidden_dim"],
            output_dim=output_dim,
            dropout_rate=checkpoint["extra_info"]["dropout_rate"],
            learning_rate=checkpoint["extra_info"]["learning_rate"],
            seed=seed,
            loss=loss,
            gamma=gamma,
            device=device,
            batch_size=batch_size,
            data_split_seed=data_split_seed,
            split_ids=split_ids,
            split_indices=split_indices,
            k_fold=k_fold,
            n_splits=n_splits,
            patience=patience,
            verbose_training=False,
            verbose_results=False,
        )

        test_results["hidden_dim"] = [
            tuple(checkpoint["extra_info"]["hidden_dim"])
            for _ in range(len(results.index))
        ]
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
        # take mean of performance on the folds
        # if k_fold=False, return performance for seed
        test_scores.append(test_results[validation_metric].mean())

    test_scores_mean = sum(test_scores) / len(test_scores)
    if verbose:
        print(f"best validation score: {save_best_model.best_valid_metric}")
        print(f"- Best model: average (test) metric score: {test_scores_mean}")
        print(f"scores for the different seeds: {test_scores}")

    if results_output is not None:
        print(
            "saving results dataframe to CSV for this "
            f"hyperparameter search in {results_output}"
        )
        results_df.to_csv(results_output)
        best_results_output = results_output.replace(".csv", "_best_model.csv")
        print(
            "saving the best model results dataframe to CSV for this "
            f"hyperparameter search in {best_results_output}"
        )
        test_results_df.to_csv(best_results_output)

    # remove any models that have been saved
    if os.path.exists(model_output):
        os.remove(model_output)

    return (
        results_df,
        test_results_df,
        save_best_model.best_valid_metric,
        checkpoint["extra_info"],
    )


def obtain_mean_history(
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    embeddings: np.array,
    path_indices: list | np.array | None = None,
    concatenate_current: bool = True,
) -> torch.tensor:
    """
    Function the obtains the mean history of the embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data
    id_column : str
        Name of the column which identifies each of the text, e.g.
        - "text_id" (if each item in `df` is a word or sentence from a particular text),
        - "user_id" (if each item in `df` is a post from a particular user),
        - "timeline_id" (if each item in `df` is a post from a particular time)
    label_column : str
        Name of the column which are corresponds to the labels of the data
    embeddings : np.array
        Corresponding embeddings for each of the items in `df`
    path_indices : list | np.array | None, optional
        The indices in the batches that we want to train and evaluate on,
        by default None. If supplied, we slice the path in this way
        before computing the mean history
    concatenate_current : bool, optional
        Whether or not to concatenate the mean history with the current
        embedding, by default True

    Returns
    -------
    torch.tensor
        Mean of the history of the embeddings of the items in `df`
    """
    paths = nlpsig.PrepareData(
        original_df=df,
        id_column=id_column,
        label_column=label_column,
        embeddings=embeddings,
    )
    # obtain column names of the embeddings in paths.df
    colnames = paths._obtain_embedding_colnames(embeddings="full")

    # initialise empty array to store mean history
    mean_history = np.zeros((len(df.index), len(colnames)))
    print("Computing the mean history for each item in the dataframe")
    for i in tqdm(range(len(df.index))):
        # look at particular text at a given index
        text = paths.df.iloc[i]
        id = text[id_column]
        timeline_index = text["timeline_index"]

        # obtain history of the particular text
        history = paths.df[
            (paths.df[id_column] == id) & (paths.df["timeline_index"] <= timeline_index)
        ][colnames]

        mean_history[i] = np.array(history).mean(axis=0)

    # slice the path in specified way
    if path_indices is not None:
        mean_history = mean_history[path_indices]
        embeddings = embeddings[path_indices]

    mean_history = mean_history.astype("float")

    # concatenate with current embedding (and convert to torch tensor)
    if concatenate_current:
        mean_history = torch.cat(
            [torch.from_numpy(mean_history), torch.from_numpy(embeddings)], dim=1
        ).float()
    else:
        mean_history = torch.from_numpy(mean_history).float()

    return mean_history


def obtain_signatures_history(
    method: str,
    dimension: int,
    sig_depth: int,
    log_signature: bool,
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    embeddings: np.array,
    seed: int = 42,
    path_indices: list | np.array | None = None,
    concatenate_current: bool = True,
) -> torch.tensor:
    """
    Function for obtaining the signature of the history of the embeddings.

    Parameters
    ----------
    method : str
        Dimension reduction method to use. See nlpsig.DimReduce for options
    dimension : int
        Dimension to reduce the embeddings to
    sig_depth : int
        Signature depth to use
    log_signature : bool
        Whether or not to use the log signature. If True, will
        compute the log signature of the history path
    df : pd.DataFrame
        Dataframe containing the data
    id_column : str
        Name of the column which identifies each of the text, e.g.
        - "text_id" (if each item in `df` is a word or sentence from a particular text),
        - "user_id" (if each item in `df` is a post from a particular user),
        - "timeline_id" (if each item in `df` is a post from a particular time)
    label_column : str
        Name of the column which are corresponds to the labels of the data
    embeddings : np.array
        Corresponding embeddings for each of the items in `df`
    seed : int, optional
        Seed to use for dimension reduction, by default 42
    path_indices : list | np.array | None, optional
        The indices in the batches that we want to train and evaluate on,
        by default None. If supplied, we slice the path in this way
        before computing the signatures
    concatenate_current : bool, optional
        Whether or not to concatenate the signatures with the current
        embedding, by default True

    Returns
    -------
    torch.tensor
        Signatures of the history of the embeddings of the items in `df`
    """
    # use nlpsig to construct the path as a numpy array
    # first define how we construct the path
    path_specifics = {
        "pad_by": "history",
        "zero_padding": True,
        "method": "max",
        "features": None,
        "embeddings": "dim_reduced",
        "include_current_embedding": True,
    }

    # first perform dimension reduction on embeddings
    if dimension == embeddings.shape[1]:
        # no need to perform dimensionality reduction
        embeddings_reduced = embeddings
    else:
        reduction = nlpsig.DimReduce(method=method, n_components=dimension)
        embeddings_reduced = reduction.fit_transform(embeddings, random_state=seed)

    # obtain path by using PrepareData class and .pad method
    paths = nlpsig.PrepareData(
        original_df=df,
        id_column=id_column,
        label_column=label_column,
        embeddings=embeddings,
        embeddings_reduced=embeddings_reduced,
    )
    path = paths.pad(**path_specifics)

    # slice the path in specified way
    if path_indices is not None:
        path = path[path_indices]
        embeddings = embeddings[path_indices]

    # remove last two columns (which contains the id and the label)
    path = path[:, :, :-2].astype("float")

    # convert to torch tensor to compute signature using signatory
    path = torch.from_numpy(path).float()
    if log_signature:
        sig = signatory.logsignature(path, sig_depth).float()
    else:
        sig = signatory.signature(path, sig_depth).float()

    # concatenate with current embedding
    if concatenate_current:
        sig = torch.cat([sig, torch.from_numpy(embeddings)], dim=1)

    return sig


def histories_baseline_hyperparameter_search(
    num_epochs: int,
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    embeddings: np.array,
    y_data: torch.tensor | np.array,
    output_dim: int,
    hidden_dim_sizes: list[list[int]] | list[int],
    dropout_rates: list[float],
    learning_rates: list[float],
    use_signatures: bool,
    seeds: list[int],
    loss: str,
    gamma: float = 0.0,
    device: str | None = None,
    batch_size: int = 64,
    log_signature: bool = False,
    dim_reduce_methods: list[str] | None = None,
    dimension_and_sig_depths: list[tuple[int, int]] | None = None,
    path_indices: list | np.array | None = None,
    data_split_seed: int = 0,
    split_ids: torch.Tensor | None = None,
    split_indices: tuple[Iterable[int] | None] | None = None,
    k_fold: bool = False,
    n_splits: int = 5,
    patience: int = 10,
    validation_metric: str = "f1",
    results_output: str | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    """
    Performs hyperparameter search for the baseline FFN model which
    concatenates the mean history (of the embeeddings) or the signature of the
    history to the current embedding, for different hidden
    for different hidden dimensions, dropout rates, learning rates
    by training and evaluating a FFN on various seeds and
    averaging performance over the seeds. If use_signatures=True,
    we also have a hyperparameter search over the dimension reduction
    methods and the dimensions and signature depths to use.

    We select the best model based on the average performance on
    the validation set. We then evaluate the best model on the test set.

    If k_fold=True, will perform k-fold validation on each seed and
    average over the average performance over the folds, otherwise
    will average over the performance of the FFNs trained using each
    seed.

    Parameters
    ----------
    num_epochs : int
        Number of epochs
    df : pd.DataFrame
        Dataframe containing the data
    id_column : str
        Name of the column which identifies each of the text, e.g.
        - "text_id" (if each item in `df` is a word or sentence from a particular text),
        - "user_id" (if each item in `df` is a post from a particular user),
        - "timeline_id" (if each item in `df` is a post from a particular time)
    label_column : str
        Name of the column which are corresponds to the labels of the data
    embeddings : np.array
        Corresponding embeddings for each of the items in `df`
    y_data : torch.tensor | np.array
        Target classification labels
    output_dim : int
        Number of unique classification labels
    hidden_dim_sizes : list[list[int]] | list[int]
        Hidden dimensions in FFN to try out. Each element in the list
        should be a list of ints if multiple hidden layers are required,
        or an int if a single hidden layer is required
    dropout_rates : list[float]
        Dropout rates to try out. Each element in the list
        should be a float
    learning_rates : list[float]
        Learning rates to try out. Each element in the list
        should be a float
    use_signatures : bool
        Whether or not to use signatures. If False, will use the mean history
        of the embeddings.
    seeds : list[int]
        Seeds to use throughout to average over the performance
        (besides for splitting the data - see data_split_seed)
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
    log_signature : bool, optional
        Whether or not to use the log signatures rather than standard signatures,
        by default False
    dim_reduce_methods : list[str] | None, optional
        Methods for dimension reduction to try out, by default None.
        Ignored if use_signatures=False. If use_signatures=True, each element
        in the list should be a string. See nlpsig.DimReduce for options.
    dimension_and_sig_depths : list[tuple[int, int]] | None, optional
        The combinations of dimensions and signature depths to use, by default None.
        Ignored if use_signatures=False. If use_signatures=True, each element
        in the list should be a tuple of ints, where the first element is the
        dimension and the second element is the corresponding signature depth.
    path_indices : list | np.array | None, optional
        The indices in the batches that we want to train and evaluate on,
        by default None. If supplied, we slice the resulting input data and target
        classification labels in this way
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
    validation_metric : str, optional
        Metric to use to use for determining the best model, by default "f1"
    results_output : str | None, optional
        Path for where to save the results dataframe, by default None
    verbose : bool, optional
        Whether or not to print out progress, by default True

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, float, dict]
        A tuple containing the full results dataframe which includes the
        performance of each of the models fitted during the hyperparameter
        search for each of the seeds, the results dataframe for the best
        performing model (based on the average validation metric performance),
        the average validation metric performance of the best model, and
        the hyperparameters which gave the best model.
    """
    if use_signatures:
        if dim_reduce_methods is None:
            msg = (
                "if use_signatures=True, must pass in the methods "
                "for dimension reduction"
            )
            raise ValueError(msg)
        if dimension_and_sig_depths is None:
            msg = (
                "if use_signatures=True, must pass in the dimensions "
                "and signature depths"
            )
            raise ValueError(msg)

    # initialise SaveBestModel class for saving the best model
    model_output = f"best_ffn_history_model_{_get_timestamp()}.pkl"
    best_model = SaveBestModel(
        metric=validation_metric, output=model_output, verbose=verbose
    )

    results_df = pd.DataFrame()
    # start model_id counter
    model_id = 0
    if use_signatures:
        for dimension, sig_depth in tqdm(dimension_and_sig_depths):
            for method in tqdm(dim_reduce_methods):
                if verbose:
                    print("\n" + "#" * 50)
                    print(
                        f"dimension: {dimension} | "
                        f"sig_depth: {sig_depth} | "
                        f"method: {method}"
                    )

                # obtain the ffn input by dimension reduction and computing signatures
                x_data = obtain_signatures_history(
                    method=method,
                    dimension=dimension,
                    sig_depth=sig_depth,
                    log_signature=log_signature,
                    df=df,
                    id_column=id_column,
                    label_column=label_column,
                    embeddings=embeddings,
                    path_indices=path_indices,
                    concatenate_current=True,
                )

                # perform hyperparameter search for FFN
                results, _, best_valid_metric, FFN_info = ffn_hyperparameter_search(
                    num_epochs=num_epochs,
                    x_data=x_data,
                    y_data=y_data,
                    output_dim=output_dim,
                    hidden_dim_sizes=hidden_dim_sizes,
                    dropout_rates=dropout_rates,
                    learning_rates=learning_rates,
                    seeds=seeds,
                    loss=loss,
                    gamma=gamma,
                    device=device,
                    batch_size=batch_size,
                    data_split_seed=data_split_seed,
                    split_ids=split_ids,
                    split_indices=split_indices,
                    k_fold=k_fold,
                    n_splits=n_splits,
                    patience=patience,
                    validation_metric=validation_metric,
                    results_output=None,
                    verbose=False,
                )

                # concatenate to results dataframe
                results["input_dim"] = x_data.shape[1]
                results["dimension"] = dimension
                results["sig_depth"] = sig_depth
                results["method"] = method
                results["log_signature"] = log_signature
                results["model_id"] = [
                    float(f"{model_id}.{id}") for id in results["model_id"]
                ]
                results_df = pd.concat([results_df, results])

                # save best model according to averaged metric over the different seeds
                # if the score is better than the previous best,
                # we save the parameters to model_output
                best_model(
                    current_valid_metric=best_valid_metric,
                    extra_info={
                        "input_dim": x_data.shape[1],
                        "dimension": dimension,
                        "sig_depth": sig_depth,
                        "method": method,
                        "log_signature": log_signature,
                        **FFN_info,
                    },
                )

                # update model_id counter
                model_id += 1
    else:
        # obtain the ffn input by averaging over history
        x_data = obtain_mean_history(
            df=df,
            id_column=id_column,
            label_column=label_column,
            embeddings=embeddings,
            path_indices=path_indices,
            concatenate_current=True,
        )

        # perform hyperparameter search for FFN
        results, _, best_valid_metric, FFN_info = ffn_hyperparameter_search(
            num_epochs=num_epochs,
            x_data=x_data,
            y_data=y_data,
            output_dim=output_dim,
            hidden_dim_sizes=hidden_dim_sizes,
            dropout_rates=dropout_rates,
            learning_rates=learning_rates,
            seeds=seeds,
            loss=loss,
            gamma=gamma,
            device=device,
            batch_size=batch_size,
            data_split_seed=data_split_seed,
            split_ids=split_ids,
            split_indices=split_indices,
            k_fold=k_fold,
            n_splits=n_splits,
            patience=patience,
            validation_metric=validation_metric,
            results_output=None,
            verbose=False,
        )

        # concatenate to results dataframe
        results["input_dim"] = x_data.shape[1]
        results["model_id"] = [float(f"{model_id}.{id}") for id in results["model_id"]]
        results_df = pd.concat([results_df, results])

        # save best model according to averaged metric over the different seeds
        # if the score is better than the previous best,
        # we save the parameters to model_output
        best_model(
            current_valid_metric=best_valid_metric,
            extra_info={"input_dim": x_data.shape[1], **FFN_info},
        )

    # load the parameters that gave the best model according to the validation metric
    checkpoint = torch.load(f=model_output)
    if verbose:
        print("*" * 50)
        print("The best model had the following parameters:")
        print(checkpoint["extra_info"])

    if use_signatures:
        # obtain the ffn input by dimension reduction and computing signatures
        x_data = obtain_signatures_history(
            method=checkpoint["extra_info"]["method"],
            dimension=checkpoint["extra_info"]["dimension"],
            sig_depth=checkpoint["extra_info"]["sig_depth"],
            log_signature=log_signature,
            df=df,
            id_column=id_column,
            label_column=label_column,
            embeddings=embeddings,
            path_indices=path_indices,
            concatenate_current=True,
        )
    else:
        # obtain the ffn input by averaging over history
        x_data = obtain_mean_history(
            df=df,
            id_column=id_column,
            label_column=label_column,
            embeddings=embeddings,
            path_indices=path_indices,
            concatenate_current=True,
        )

    # implement model again and obtain the results dataframe
    # and evaluate on the test set
    test_scores = []
    test_results_df = pd.DataFrame()
    for seed in seeds:
        _, test_results = implement_ffn(
            num_epochs=num_epochs,
            x_data=x_data,
            y_data=y_data,
            output_dim=output_dim,
            hidden_dim=checkpoint["extra_info"]["hidden_dim"],
            dropout_rate=checkpoint["extra_info"]["dropout_rate"],
            learning_rate=checkpoint["extra_info"]["learning_rate"],
            seed=seed,
            loss=loss,
            gamma=gamma,
            device=device,
            batch_size=batch_size,
            data_split_seed=data_split_seed,
            split_ids=split_ids,
            split_indices=split_indices,
            k_fold=k_fold,
            n_splits=n_splits,
            patience=patience,
            verbose_training=False,
            verbose_results=False,
        )

        test_results["hidden_dim"] = [
            tuple(checkpoint["extra_info"]["hidden_dim"])
            for _ in range(len(test_results.index))
        ]
        test_results["dropout_rate"] = checkpoint["extra_info"]["dropout_rate"]
        test_results["learning_rate"] = checkpoint["extra_info"]["learning_rate"]
        test_results["seed"] = seed
        test_results["loss_function"] = loss
        test_results["gamma"] = gamma
        test_results["k_fold"] = k_fold
        test_results["n_splits"] = n_splits if k_fold else None
        test_results["batch_size"] = batch_size
        test_results["input_dim"] = checkpoint["extra_info"]["input_dim"]
        if use_signatures:
            test_results["dimension"] = checkpoint["extra_info"]["dimension"]
            test_results["sig_depth"] = checkpoint["extra_info"]["sig_depth"]
            test_results["method"] = checkpoint["extra_info"]["method"]
            test_results["log_signature"] = checkpoint["extra_info"]["log_signature"]
        test_results_df = pd.concat([test_results_df, test_results])

        # save metric that we want to validate on
        # take mean of performance on the folds
        # if k_fold=False, return performance for seed
        test_scores.append(test_results[validation_metric].mean())

    test_scores_mean = sum(test_scores) / len(test_scores)

    if verbose:
        print(f"best validation score: {best_model.best_valid_metric}")
        print(f"- Best model: average (test) metric score: {test_scores_mean}")
        print(f"scores for the different seeds: {test_scores}")

    if results_output is not None:
        print(
            "saving results dataframe to CSV for this "
            f"hyperparameter search in {results_output}"
        )
        results_df.to_csv(results_output)
        best_results_output = results_output.replace(".csv", "_best_model.csv")
        print(
            "saving the best model results dataframe to CSV for this "
            f"hyperparameter search in {best_results_output}"
        )
        test_results_df.to_csv(best_results_output)

    # remove any models that have been saved
    if os.path.exists(model_output):
        os.remove(model_output)

    return (
        results_df,
        test_results_df,
        best_model.best_valid_metric,
        checkpoint["extra_info"],
    )
