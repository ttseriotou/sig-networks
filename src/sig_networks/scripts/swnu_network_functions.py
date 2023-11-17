from __future__ import annotations

import os
from typing import Iterable

import nlpsig
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from sig_networks.pytorch_utils import SaveBestModel, _get_timestamp, set_seed
from sig_networks.scripts.implement_model import implement_model
from sig_networks.swnu_network import SWNUNetwork


def obtain_SWNUNetwork_input(
    method: str,
    dimension: int,
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    embeddings: np.array,
    k: int,
    features: list[str] | str | None = None,
    standardise_method: list[str] | str | None = None,
    include_features_in_path: bool = False,
    include_features_in_input: bool = False,
    seed: int = 42,
    path_indices: list | np.array | None = None,
) -> dict[str, torch.tensor | int]:
    """
    Function to obtain the input for a SW-unit model
    (e.g. SWNUNetwork or SWMHAUNetwork).

    Parameters
    ----------
    method : str
        Dimension reduction method to use. See nlpsig.DimReduce for options
    dimension : int
        Dimension to reduce the embeddings to
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
    k : int
        The history length to use
    features : list[str] | str | None, optional
        Which feature(s) to keep. If None, then doesn't keep any.
    standardise_method : list[str] | str | None, optional
        If not None, applies standardisation to the features, default None.
        If a list is passed, must be the same length as `features`.
        See nlpsig.PrepareData.pad() for options.
    include_features_in_path : bool
        Whether or not to keep the additional features
        (e.g. time features) within the path.
    include_features_in_input : bool
        Whether or not to concatenate the additional features into the FFN
    seed : int, optional
        Seed to use for dimension reduction, by default 42
    path_indices : list | np.array | None, optional
        The indices in the batches that we want to train and evaluate on,
        by default None. If supplied, we slice the path in this way

    Returns
    -------
    dict[str, torch.tensor | int]
        Dictionary where:
        - "x_data" is a dictionary where:
            - "path" is a tensor of the path to be passed
              into the SeqSigNet network
            - "features" is a tensor of the features
              (e.g. time features or additional features)
        - "input_channels" is the number of channels in the path
        - "num_features" is the number of features
          (e.g. time features or additional features)
          (this is None if there are no additional features to concatenate)
    """
    # use nlpsig to construct the path as a numpy array
    # first define how we construct the path
    path_specifics = {
        "pad_by": "history",
        "zero_padding": True,
        "method": "k_last",
        "k": k,
        "features": features,
        "standardise_method": standardise_method,
        "embeddings": "dim_reduced",
        "include_current_embedding": True,
        "pad_from_below": True,
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
    paths.pad(**path_specifics)

    # construct path for SWNUNetwork which is given as a dictionary with keys
    # "x_data", "input_channels" and "num_features"
    # include features and (full, not dimension reduced) embeddings in the FFN input
    return paths.get_torch_path_for_SWNUNetwork(
        include_features_in_path=include_features_in_path,
        include_features_in_input=include_features_in_input,
        include_embedding_in_input=True,
        reduced_embeddings=False,
        path_indices=path_indices,
    )


def implement_swnu_network(
    num_epochs: int,
    x_data: dict[str, np.array | torch.Tensor],
    y_data: torch.tensor | np.array,
    input_channels: int,
    num_features: int,
    embedding_dim: int,
    log_signature: bool,
    sig_depth: int,
    pooling: str,
    swnu_hidden_dim: list[int] | int,
    ffn_hidden_dim: list[int] | int,
    output_dim: int,
    BiLSTM: bool,
    dropout_rate: float,
    learning_rate: float,
    seed: int,
    loss: str,
    gamma: float = 0.0,
    device: str | None = None,
    batch_size: int = 64,
    output_channels: int | None = None,
    augmentation_type: str = "Conv1d",
    hidden_dim_aug: list[int] | int | None = None,
    comb_method: str = "concatenation",
    data_split_seed: int = 0,
    split_ids: torch.Tensor | None = None,
    split_indices: tuple[Iterable[int] | None] | None = None,
    k_fold: bool = False,
    n_splits: int = 5,
    patience: int = 10,
    verbose_training: bool = False,
    verbose_results: bool = False,
    verbose_model: bool = False,
) -> tuple[SWNUNetwork, pd.DataFrame]:
    """
    Function which takes in input variables, x_data,
    and target output classification labels, y_data,
    and train and evaluates a SWNUNetwork model.

    If k_fold=True, it will evaluate the SWNUNetwork by performing
    k-fold validation with n_splits number of folds (i.e. train and test
    n_split number of SWNUNetwork models), otherwise, it will
    evaluate by training and testing a single SWNUNetwork on one
    particular split of the data.

    Parameters
    ----------
    num_epochs : int
        Number of epochs
    x_data : dict[str, np.array | torch.Tensor]
        A dictionary containing the input data. The keys should be
        "path" and "features"
    y_data : torch.tensor | np.array
        Target classification labels
    input_channels : int
        Dimension of the embeddings in the path that will be passed in.
    num_features : int
        Number of time features to add to FFN input. If none, set to zero.
    embedding_dim: int
        Dimensions of current BERT post embedding. Usually 384 or 768.
    log_signature : bool
        Whether or not to use the log signature or standard signature.
    sig_depth : int
        The depth to truncate the path signature at.
    pooling: str | None
        Pooling operation to apply. If None, no pooling is applied.
        Options are:
            - "signature": apply signature on the LSTM units at the end
                to obtain the final history representation
            - "lstm": take the final (non-padded) LSTM unit as the final
                history representation
    swnu_hidden_dim : list[int] | int
        Dimensions of the hidden layers in the SNWU blocks, can be int if a single
        SWNU block, or can be a list of ints for multiple SWNU blocks
    output_dim : int
        Number of unique classification labels
    BiLSTM : bool
        Whether or not to use a BiLSTM in the SWNU units
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
    output_channels : int | None, optional
        Requested dimension of the embeddings after convolution layer.
        If None, will be set to the last item in `swnu_hidden_dim`, by default None.
    augmentation_type : str, optional
        Method of augmenting the path, by default "Conv1d".
        Options are:
        - "Conv1d": passes path through 1D convolution layer.
        - "signatory": passes path through `Augment` layer from `signatory` package.
    hidden_dim_aug : list[int] | int | None
        Dimensions of the hidden layers in the augmentation layer.
        Passed into `Augment` class from `signatory` package if
        `augmentation_type='signatory'`, by default None.
    comb_method : str, optional
        Determines how to combine the path signature and embeddings,
        by default "gated_addition".
        Options are:
        - concatenation: concatenation of path signature and embedding vector
        - gated_addition: element-wise addition of path signature
            and embedding vector
        - gated_concatenation: concatenation of linearly gated path signature
            and embedding vector
        - scaled_concatenation: concatenation of single value scaled path
            signature and embedding vector
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
    tuple[SWNUNetwork, pd.DataFrame]
        SWNUNetwork object (if k-fold, this is a randomly
        initialised model, otherwise it has been trained on the data splits
        that were generated within this function with data_split_seed),
        and dataframe of the evaluation metrics for the validation and
        test sets generated within this function.
    """
    # set seed
    set_seed(seed)

    # initialise SWNUNetwork
    SWNUNetwork_args = {
        "input_channels": input_channels,
        "num_features": num_features,
        "embedding_dim": embedding_dim,
        "log_signature": log_signature,
        "sig_depth": sig_depth,
        "pooling": pooling,
        "hidden_dim_swnu": swnu_hidden_dim,
        "hidden_dim_ffn": ffn_hidden_dim,
        "output_dim": output_dim,
        "dropout_rate": dropout_rate,
        "output_channels": output_channels,
        "augmentation_type": augmentation_type,
        "hidden_dim_aug": hidden_dim_aug,
        "BiLSTM": BiLSTM,
        "comb_method": comb_method,
    }
    swnu_network_model = SWNUNetwork(**SWNUNetwork_args)

    if verbose_model:
        print(swnu_network_model)

    # convert data to torch tensors
    # deal with case if x_data is a dictionary
    if isinstance(x_data, dict):
        # iterate through the values and check they are of the correct type
        for key, value in x_data.items():
            if not isinstance(value, torch.Tensor):
                x_data[key] = torch.tensor(value)
            x_data[key] = x_data[key].float()
    else:
        # convert data to torch tensors
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data).float()
        x_data = x_data.float()
    if not isinstance(y_data, torch.Tensor):
        y_data = torch.tensor(y_data)

    return implement_model(
        model=swnu_network_model,
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


def swnu_network_hyperparameter_search(
    num_epochs: int,
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    embeddings: np.array,
    y_data: np.array,
    output_dim: int,
    history_lengths: list[int],
    dim_reduce_methods: list[str],
    dimensions: list[int],
    log_signature: bool,
    pooling: str,
    swnu_hidden_dim_sizes_and_sig_depths: list[tuple[list[int] | int, int]],
    ffn_hidden_dim_sizes: list[int] | list[list[int]],
    dropout_rates: list[float],
    learning_rates: list[float],
    BiLSTM: bool,
    seeds: list[int],
    loss: str,
    gamma: float = 0.0,
    device: str | None = None,
    batch_size: int = 64,
    features: list[str] | str | None = None,
    standardise_method: list[str] | str | None = None,
    include_features_in_path: bool = False,
    include_features_in_input: bool = False,
    conv_output_channels: list[int] | None = None,
    augmentation_type: str = "Conv1d",
    hidden_dim_aug: list[int] | int | None = None,
    comb_method: str = "concatenation",
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
    Performs hyperparameter search for the SWNUNetwork model for different
    SWNU hidden dimensions and sig depths, LSTM hidden dimensions,
    FFN hidden dimensions, dropout rates, learning rates by training and
    evaluating a SWNUNetworks on various seeds and averaging performance
    over the seeds.

    If k_fold=True, will perform k-fold validation on each seed and
    average over the average performance over the folds, otherwise
    will average over the performance of the SWNUNetwork
    trained using each seed.

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
    history_lengths : list[int]
        A list of history lengths to use
    dim_reduce_methods : list[str]
        Methods for dimension reduction to try out.
        Each element in the list should be a string.
        See nlpsig.DimReduce for options.
    dimensions : list[int]
        Dimensions to reduce to
    log_signature : bool
        Whether or not to use the log signatures rather than standard signatures
    pooling: str | None
        Pooling operation to apply. If None, no pooling is applied.
        Options are:
            - "signature": apply signature on the LSTM units at the end
                to obtain the final history representation
            - "lstm": take the final (non-padded) LSTM unit as the final
                history representation
    swnu_hidden_dim_sizes_and_sig_depths : list[tuple[list[int] | int, int]]
        A list of tuples, where each tuple contains the hidden dimensions
        for the SWNU blocks and the signature depth to take. The hidden
        dimensions can be a list of ints if multiple SWNU blocks, or can
        be an int if a single SWNU block
    ffn_hidden_dim_sizes : list[int] | list[list[int]]
        Hidden dimensions in FFN to try out. Each element in the list
        should be a list of ints if multiple hidden layers are required,
        or an int if a single hidden layer is required
    dropout_rates : list[float]
        Dropout rates to try out. Each element in the list
        should be a float
    learning_rates : list[float]
        Learning rates to try out. Each element in the list
        should be a float
    BiLSTM : bool
        Whether or not to use a BiLSTM in the SWNU units
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
    features : list[str] | str | None, optional
            Which feature(s) to keep. If None, then doesn't keep any.
    standardise_method : list[str] | str | None, optional
        If not None, applies standardisation to the features, default None.
        If a list is passed, must be the same length as `features`.
        See nlpsig.PrepareData.pad() for options.
    include_features_in_path : bool
        Whether or not to keep the additional features
        (e.g. time features) within the path.
    include_features_in_input : bool
        Whether or not to concatenate the additional features into the FFN
    conv_output_channels : list[int] | None, optional
        List of requested dimensions of the embeddings after convolution layer.
        If None, will be set to the last item in `swnu_hidden_dim`, by default None.
    augmentation_type : str, optional
        Method of augmenting the path, by default "Conv1d".
        Options are:
        - "Conv1d": passes path through 1D convolution layer.
        - "signatory": passes path through `Augment` layer from `signatory` package.
    hidden_dim_aug : list[int] | int | None
        Dimensions of the hidden layers in the augmentation layer.
        Passed into `Augment` class from `signatory` package if
        `augmentation_type='signatory'`, by default None.
    comb_method : str, optional
        Determines how to combine the path signature and embeddings,
        by default "gated_addition".
        Options are:
        - concatenation: concatenation of path signature and embedding vector
        - gated_addition: element-wise addition of path signature
            and embedding vector
        - gated_concatenation: concatenation of linearly gated path signature
            and embedding vector
        - scaled_concatenation: concatenation of single value scaled path
            signature and embedding vector
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
    if validation_metric not in ["loss", "accuracy", "f1"]:
        raise ValueError("validation_metric must be either 'loss', 'accuracy' or 'f1'")

    # initialise SaveBestModel class
    model_output = f"best_swnu_network_model_{_get_timestamp()}.pkl"
    best_model = SaveBestModel(
        metric=validation_metric, output=model_output, verbose=verbose
    )

    if isinstance(features, str):
        features = [features]
    elif features is None:
        features = []

    if isinstance(standardise_method, str):
        standardise_method = [standardise_method]
    elif standardise_method is None:
        standardise_method = []

    if conv_output_channels is None:
        conv_output_channels = [None]

    # find model parameters that has the best validation
    results_df = pd.DataFrame()
    # start model_id counter
    model_id = 0
    for k in tqdm(history_lengths):
        if verbose:
            print("\n" + "-" * 50)
            print(f"k: {k}")
        for dimension in tqdm(dimensions):
            for method in tqdm(dim_reduce_methods):
                print("\n" + "#" * 50)
                print(f"dimension: {dimension} | " f"method: {method}")
                input = obtain_SWNUNetwork_input(
                    method=method,
                    dimension=dimension,
                    df=df,
                    id_column=id_column,
                    label_column=label_column,
                    embeddings=embeddings,
                    k=k,
                    features=features,
                    standardise_method=standardise_method,
                    include_features_in_path=include_features_in_path,
                    include_features_in_input=include_features_in_input,
                    path_indices=path_indices,
                )

                for swnu_hidden_dim, sig_depth in tqdm(
                    swnu_hidden_dim_sizes_and_sig_depths
                ):
                    for ffn_hidden_dim in tqdm(ffn_hidden_dim_sizes):
                        for output_channels in tqdm(conv_output_channels):
                            for dropout in tqdm(dropout_rates):
                                for lr in tqdm(learning_rates):
                                    if verbose:
                                        print("\n" + "!" * 50)
                                        print(
                                            f"swnu_hidden_dim: {swnu_hidden_dim} | "
                                            f"ffn_hidden_dim: {ffn_hidden_dim} | "
                                            f"sig_depth: {sig_depth} | "
                                            f"output_channels: {output_channels} | "
                                            f"dropout: {dropout} | "
                                            f"learning_rate: {lr}"
                                        )

                                    scores = []
                                    verbose_model = verbose
                                    for seed in seeds:
                                        _, results = implement_swnu_network(
                                            num_epochs=num_epochs,
                                            x_data=input["x_data"],
                                            y_data=y_data,
                                            input_channels=input["input_channels"],
                                            output_channels=output_channels,
                                            num_features=input["num_features"],
                                            embedding_dim=input["embedding_dim"],
                                            log_signature=log_signature,
                                            sig_depth=sig_depth,
                                            pooling=pooling,
                                            swnu_hidden_dim=swnu_hidden_dim,
                                            ffn_hidden_dim=ffn_hidden_dim,
                                            output_dim=output_dim,
                                            BiLSTM=BiLSTM,
                                            dropout_rate=dropout,
                                            learning_rate=lr,
                                            seed=seed,
                                            loss=loss,
                                            gamma=gamma,
                                            device=device,
                                            batch_size=batch_size,
                                            augmentation_type=augmentation_type,
                                            hidden_dim_aug=hidden_dim_aug,
                                            comb_method=comb_method,
                                            data_split_seed=data_split_seed,
                                            split_ids=split_ids,
                                            split_indices=split_indices,
                                            k_fold=k_fold,
                                            n_splits=n_splits,
                                            patience=patience,
                                            verbose_results=verbose,
                                            verbose_model=verbose_model,
                                        )
                                        # save metric that we want to validate on
                                        # take mean of performance on the folds
                                        # if k_fold=False, return performance for seed
                                        scores.append(
                                            results[f"valid_{validation_metric}"].mean()
                                        )

                                        # concatenate to results dataframe
                                        results["k"] = k
                                        results["dimensions"] = dimension
                                        results["sig_depth"] = sig_depth
                                        results["method"] = method
                                        results["input_channels"] = input[
                                            "input_channels"
                                        ]
                                        results["output_channels"] = output_channels
                                        results["features"] = [features]
                                        results["standardise_method"] = [
                                            standardise_method
                                        ]
                                        results[
                                            "include_features_in_path"
                                        ] = include_features_in_path
                                        results[
                                            "include_features_in_input"
                                        ] = include_features_in_input
                                        results["embedding_dim"] = input[
                                            "embedding_dim"
                                        ]
                                        results["num_features"] = input["num_features"]
                                        results["log_signature"] = log_signature
                                        results["pooling"] = pooling
                                        results["swnu_hidden_dim"] = [
                                            tuple(swnu_hidden_dim)
                                            for _ in range(len(results.index))
                                        ]
                                        results["ffn_hidden_dim"] = [
                                            tuple(ffn_hidden_dim)
                                            for _ in range(len(results.index))
                                        ]
                                        results["dropout_rate"] = dropout
                                        results["learning_rate"] = lr
                                        results["seed"] = seed
                                        results["BiLSTM"] = BiLSTM
                                        results["loss_function"] = loss
                                        results["gamma"] = gamma
                                        results["k_fold"] = k_fold
                                        results["n_splits"] = (
                                            n_splits if k_fold else None
                                        )
                                        results["augmentation_type"] = augmentation_type
                                        results["hidden_dim_aug"] = (
                                            [
                                                tuple(hidden_dim_aug)
                                                for _ in range(len(results.index))
                                            ]
                                            if hidden_dim_aug is not None
                                            else None
                                        )
                                        results["comb_method"] = comb_method
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
                                        print(
                                            f"scores for the different seeds: {scores}"
                                        )

                                    # save best model according to averaged metric
                                    # over the different seeds
                                    # if the score is better than the previous best,
                                    # we save the parameters to model_output
                                    best_model(
                                        current_valid_metric=scores_mean,
                                        extra_info={
                                            "k": k,
                                            "dimensions": dimension,
                                            "sig_depth": sig_depth,
                                            "method": method,
                                            "input_channels": input["input_channels"],
                                            "output_channels": output_channels,
                                            "features": features,
                                            "standardise_method": standardise_method,
                                            "include_features_in_path": (
                                                include_features_in_path
                                            ),
                                            "include_features_in_input": (
                                                include_features_in_input
                                            ),
                                            "embedding_dim": input["embedding_dim"],
                                            "num_features": input["num_features"],
                                            "log_signature": log_signature,
                                            "pooling": pooling,
                                            "swnu_hidden_dim": swnu_hidden_dim,
                                            "ffn_hidden_dim": ffn_hidden_dim,
                                            "dropout_rate": dropout,
                                            "learning_rate": lr,
                                            "BiLSTM": BiLSTM,
                                            "augmentation_type": augmentation_type,
                                            "hidden_dim_aug": hidden_dim_aug,
                                            "comb_method": comb_method,
                                        },
                                    )

                                    # update model_id counter
                                    model_id += 1

    checkpoint = torch.load(f=model_output)
    if verbose:
        print("*" * 50)
        print("The best model had the following parameters:")
        print(checkpoint["extra_info"])

    input = obtain_SWNUNetwork_input(
        method=checkpoint["extra_info"]["method"],
        dimension=checkpoint["extra_info"]["dimensions"],
        df=df,
        id_column=id_column,
        label_column=label_column,
        embeddings=embeddings,
        k=checkpoint["extra_info"]["k"],
        features=checkpoint["extra_info"]["features"],
        standardise_method=checkpoint["extra_info"]["standardise_method"],
        include_features_in_path=checkpoint["extra_info"]["include_features_in_path"],
        include_features_in_input=checkpoint["extra_info"]["include_features_in_input"],
        path_indices=path_indices,
    )

    # implement model again and obtain the results dataframe
    # and evaluate on the test set
    test_scores = []
    test_results_df = pd.DataFrame()
    for seed in seeds:
        _, test_results = implement_swnu_network(
            num_epochs=num_epochs,
            x_data=input["x_data"],
            y_data=y_data,
            input_channels=input["input_channels"],
            output_channels=checkpoint["extra_info"]["output_channels"],
            num_features=input["num_features"],
            embedding_dim=input["embedding_dim"],
            log_signature=checkpoint["extra_info"]["log_signature"],
            sig_depth=checkpoint["extra_info"]["sig_depth"],
            pooling=checkpoint["extra_info"]["pooling"],
            swnu_hidden_dim=checkpoint["extra_info"]["swnu_hidden_dim"],
            ffn_hidden_dim=checkpoint["extra_info"]["ffn_hidden_dim"],
            output_dim=output_dim,
            BiLSTM=checkpoint["extra_info"]["BiLSTM"],
            dropout_rate=checkpoint["extra_info"]["dropout_rate"],
            learning_rate=checkpoint["extra_info"]["learning_rate"],
            seed=seed,
            loss=loss,
            gamma=gamma,
            device=device,
            batch_size=batch_size,
            augmentation_type=checkpoint["extra_info"]["augmentation_type"],
            hidden_dim_aug=checkpoint["extra_info"]["hidden_dim_aug"],
            comb_method=checkpoint["extra_info"]["comb_method"],
            data_split_seed=data_split_seed,
            split_ids=split_ids,
            split_indices=split_indices,
            k_fold=k_fold,
            n_splits=n_splits,
            patience=patience,
            verbose_training=False,
            verbose_results=False,
            verbose_model=False,
        )

        # save metric that we want to validate on
        # taking the mean over the performance on the folds for the seed
        # if k_fold=False, this returns the performance for the seed
        test_scores.append(test_results[validation_metric].mean())

        # concatenate to results dataframe
        test_results["k"] = checkpoint["extra_info"]["k"]
        test_results["dimensions"] = checkpoint["extra_info"]["dimensions"]
        test_results["sig_depth"] = checkpoint["extra_info"]["sig_depth"]
        test_results["method"] = checkpoint["extra_info"]["method"]
        test_results["input_channels"] = input["input_channels"]
        test_results["output_channels"] = checkpoint["extra_info"]["output_channels"]
        test_results["features"] = [features]
        test_results["standardise_method"] = [standardise_method]
        test_results["include_features_in_path"] = include_features_in_path
        test_results["include_features_in_input"] = include_features_in_input
        test_results["embedding_dim"] = input["embedding_dim"]
        test_results["num_features"] = input["num_features"]
        test_results["log_signature"] = checkpoint["extra_info"]["log_signature"]
        test_results["pooling"] = checkpoint["extra_info"]["pooling"]
        test_results["swnu_hidden_dim"] = [
            tuple(checkpoint["extra_info"]["swnu_hidden_dim"])
            for _ in range(len(test_results.index))
        ]
        test_results["ffn_hidden_dim"] = [
            tuple(checkpoint["extra_info"]["ffn_hidden_dim"])
            for _ in range(len(test_results.index))
        ]
        test_results["dropout_rate"] = checkpoint["extra_info"]["dropout_rate"]
        test_results["learning_rate"] = checkpoint["extra_info"]["learning_rate"]
        test_results["seed"] = seed
        test_results["BiLSTM"] = checkpoint["extra_info"]["BiLSTM"]
        test_results["loss_function"] = loss
        test_results["gamma"] = gamma
        test_results["k_fold"] = k_fold
        test_results["n_splits"] = n_splits if k_fold else None
        test_results["augmentation_type"] = checkpoint["extra_info"][
            "augmentation_type"
        ]
        test_results["hidden_dim_aug"] = (
            None
            if checkpoint["extra_info"]["hidden_dim_aug"] is None
            else [
                (checkpoint["extra_info"]["hidden_dim_aug"],)
                for _ in range(len(test_results.index))
            ]
            if (type(checkpoint["extra_info"]["hidden_dim_aug"]) == int)
            else [
                tuple(checkpoint["extra_info"]["hidden_dim_aug"])
                for _ in range(len(test_results.index))
            ]
        )
        test_results["comb_method"] = checkpoint["extra_info"]["comb_method"]
        test_results["batch_size"] = batch_size
        test_results_df = pd.concat([test_results_df, test_results])

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
