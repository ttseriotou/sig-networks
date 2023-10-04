from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from nlpsig_networks.pytorch_utils import SaveBestModel, _get_timestamp, set_seed
from nlpsig_networks.scripts.implement_model import implement_model
from nlpsig_networks.scripts.seqsignet_functions import obtain_SeqSigNet_input
from nlpsig_networks.seqsignet_attention_bilstm import SeqSigNetAttentionBiLSTM


def implement_seqsignet_attention_bilstm(
    num_epochs: int,
    x_data: np.array | torch.Tensor | dict[str, np.array | torch.Tensor],
    y_data: torch.tensor | np.array,
    input_channels: int,
    output_channels: int,
    num_features: int,
    embedding_dim: int,
    log_signature: bool,
    sig_depth: int,
    pooling: str,
    num_heads: int,
    num_layers: int,
    lstm_hidden_dim: int,
    ffn_hidden_dim: list[int] | int,
    output_dim: int,
    dropout_rate: float,
    learning_rate: float,
    seed: int,
    loss: str,
    gamma: float = 0.0,
    device: str | None = None,
    batch_size: int = 64,
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
) -> tuple[SeqSigNetAttentionBiLSTM, pd.DataFrame]:
    # set seed
    set_seed(seed)

    # initialise SeqSigNetAttentionBiLSTM
    SeqSigNetAttentionBiLSTM_args = {
        "input_channels": input_channels,
        "output_channels": output_channels,
        "num_features": num_features,
        "embedding_dim": embedding_dim,
        "log_signature": log_signature,
        "sig_depth": sig_depth,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "hidden_dim_lstm": lstm_hidden_dim,
        "hidden_dim_ffn": ffn_hidden_dim,
        "output_dim": output_dim,
        "dropout_rate": dropout_rate,
        "pooling": pooling,
        "augmentation_type": augmentation_type,
        "hidden_dim_aug": hidden_dim_aug,
        "comb_method": comb_method,
    }
    seqsignet_attention_bilstm_model = SeqSigNetAttentionBiLSTM(
        **SeqSigNetAttentionBiLSTM_args
    )

    if verbose_model:
        print(seqsignet_attention_bilstm_model)

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
        model=seqsignet_attention_bilstm_model,
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


def seqsignet_attention_bilstm_hyperparameter_search(
    num_epochs: int,
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    embeddings: np.array,
    y_data: np.array,
    output_dim: int,
    shift: int,
    window_size: int,
    n: int,
    dim_reduce_methods: list[str],
    dimensions: list[int],
    log_signature: bool,
    pooling: str,
    swmhau_parameters: list[tuple[int, int, int]],
    num_layers: list[int],
    lstm_hidden_dim_sizes: list[int],
    ffn_hidden_dim_sizes: list[int] | list[list[int]],
    dropout_rates: list[float],
    learning_rates: list[float],
    seeds: list[int],
    loss: str,
    gamma: float = 0.0,
    device: str | None = None,
    batch_size: int = 64,
    features: list[str] | str | None = None,
    standardise_method: list[str] | str | None = None,
    include_features_in_path: bool = False,
    include_features_in_input: bool = False,
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
):
    if validation_metric not in ["loss", "accuracy", "f1"]:
        raise ValueError("validation_metric must be either 'loss', 'accuracy' or 'f1'")

    # initialise SaveBestModel class
    model_output = f"best_seqsignet_attention_bilstm_model_{_get_timestamp()}.pkl"
    save_best_model = SaveBestModel(
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

    # find model parameters that has the best validation
    results_df = pd.DataFrame()
    model_id = 0
    k = shift * n + (window_size - shift)
    for dimension in tqdm(dimensions):
        for method in tqdm(dim_reduce_methods):
            print("\n" + "#" * 50)
            print(f"dimension: {dimension} | " f"method: {method}")
            input = obtain_SeqSigNet_input(
                method=method,
                dimension=dimension,
                df=df,
                id_column=id_column,
                label_column=label_column,
                embeddings=embeddings,
                shift=shift,
                window_size=window_size,
                n=n,
                features=features,
                standardise_method=standardise_method,
                include_features_in_path=include_features_in_path,
                include_features_in_input=include_features_in_input,
                path_indices=path_indices,
            )

            for output_channels, sig_depth, num_heads in tqdm(swmhau_parameters):
                for n_layers in tqdm(num_layers):
                    for lstm_hidden_dim in tqdm(lstm_hidden_dim_sizes):
                        for ffn_hidden_dim in tqdm(ffn_hidden_dim_sizes):
                            for dropout in tqdm(dropout_rates):
                                for lr in tqdm(learning_rates):
                                    if verbose:
                                        print("\n" + "!" * 50)
                                        print(
                                            f"output_channels: {output_channels} | "
                                            f"lstm_hidden_dim: {lstm_hidden_dim} |"
                                            f"ffn_hidden_dim: {ffn_hidden_dim} | "
                                            f"sig_depth: {sig_depth} | "
                                            f"num_heads: {num_heads} | "
                                            f"dropout: {dropout} | "
                                            f"learning_rate: {lr}"
                                        )

                                    scores = []
                                    verbose_model = verbose
                                    for seed in seeds:
                                        (
                                            _,
                                            results,
                                        ) = implement_seqsignet_attention_bilstm(
                                            num_epochs=num_epochs,
                                            x_data=input["x_data"],
                                            y_data=y_data,
                                            input_channels=input["input_channels"],
                                            output_channels=output_channels,
                                            embedding_dim=input["embedding_dim"],
                                            num_features=input["num_features"],
                                            log_signature=log_signature,
                                            sig_depth=sig_depth,
                                            pooling=pooling,
                                            num_heads=num_heads,
                                            num_layers=n_layers,
                                            lstm_hidden_dim=lstm_hidden_dim,
                                            ffn_hidden_dim=ffn_hidden_dim,
                                            output_dim=output_dim,
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
                                            verbose_training=False,
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
                                        results["shift"] = shift
                                        results["window_size"] = window_size
                                        results["n"] = n
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
                                        results["num_heads"] = num_heads
                                        results["num_layers"] = n_layers
                                        results["lstm_hidden_dim"] = lstm_hidden_dim
                                        results["ffn_hidden_dim"] = [
                                            tuple(ffn_hidden_dim)
                                            for _ in range(len(results.index))
                                        ]
                                        results["dropout_rate"] = dropout
                                        results["learning_rate"] = lr
                                        results["seed"] = seed
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

                                    model_id += 1
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
                                    save_best_model(
                                        current_valid_metric=scores_mean,
                                        extra_info={
                                            "k": k,
                                            "shift": shift,
                                            "window_size": window_size,
                                            "n": n,
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
                                            "num_heads": num_heads,
                                            "num_layers": n_layers,
                                            "lstm_hidden_dim": lstm_hidden_dim,
                                            "ffn_hidden_dim": ffn_hidden_dim,
                                            "dropout_rate": dropout,
                                            "learning_rate": lr,
                                            "augmentation_type": augmentation_type,
                                            "hidden_dim_aug": hidden_dim_aug,
                                            "comb_method": comb_method,
                                        },
                                    )

    checkpoint = torch.load(f=model_output)
    if verbose:
        print("*" * 50)
        print("The best model had the following parameters:")
        print(checkpoint["extra_info"])

    input = obtain_SeqSigNet_input(
        method=checkpoint["extra_info"]["method"],
        dimension=checkpoint["extra_info"]["dimensions"],
        df=df,
        id_column=id_column,
        label_column=label_column,
        embeddings=embeddings,
        shift=checkpoint["extra_info"]["shift"],
        window_size=checkpoint["extra_info"]["window_size"],
        n=checkpoint["extra_info"]["n"],
        features=checkpoint["extra_info"]["features"],
        standardise_method=checkpoint["extra_info"]["standardise_method"],
        include_features_in_path=checkpoint["extra_info"]["include_features_in_path"],
        include_features_in_input=checkpoint["extra_info"]["include_features_in_input"],
        path_indices=path_indices,
    )

    test_scores = []
    test_results_df = pd.DataFrame()
    for seed in seeds:
        _, test_results = implement_seqsignet_attention_bilstm(
            num_epochs=num_epochs,
            x_data=input["x_data"],
            y_data=y_data,
            input_channels=checkpoint["extra_info"]["input_channels"],
            output_channels=checkpoint["extra_info"]["output_channels"],
            embedding_dim=input["embedding_dim"],
            num_features=input["num_features"],
            log_signature=checkpoint["extra_info"]["log_signature"],
            sig_depth=checkpoint["extra_info"]["sig_depth"],
            pooling=checkpoint["extra_info"]["pooling"],
            num_heads=checkpoint["extra_info"]["num_heads"],
            num_layers=checkpoint["extra_info"]["num_layers"],
            lstm_hidden_dim=checkpoint["extra_info"]["lstm_hidden_dim"],
            ffn_hidden_dim=checkpoint["extra_info"]["ffn_hidden_dim"],
            output_dim=output_dim,
            dropout_rate=checkpoint["extra_info"]["dropout_rate"],
            learning_rate=checkpoint["extra_info"]["learning_rate"],
            seed=seed,
            loss=loss,
            gamma=gamma,
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
        # take mean of performance on the folds
        # if k_fold=False, return performance for seed
        test_scores.append(test_results[validation_metric].mean())

        # concatenate to results dataframe
        test_results["k"] = checkpoint["extra_info"]["k"]
        test_results["shift"] = checkpoint["extra_info"]["shift"]
        test_results["window_size"] = checkpoint["extra_info"]["window_size"]
        test_results["n"] = checkpoint["extra_info"]["n"]
        test_results["dimensions"] = checkpoint["extra_info"]["dimensions"]
        test_results["sig_depth"] = checkpoint["extra_info"]["sig_depth"]
        test_results["method"] = checkpoint["extra_info"]["method"]
        test_results["input_channels"] = checkpoint["extra_info"]["input_channels"]
        test_results["output_channels"] = checkpoint["extra_info"]["output_channels"]
        test_results["features"] = [features]
        test_results["standardise_method"] = [standardise_method]
        test_results["include_features_in_path"] = include_features_in_path
        test_results["include_features_in_input"] = include_features_in_input
        test_results["embedding_dim"] = input["embedding_dim"]
        test_results["num_features"] = input["num_features"]
        test_results["log_signature"] = checkpoint["extra_info"]["log_signature"]
        test_results["pooling"] = checkpoint["extra_info"]["pooling"]
        test_results["num_heads"] = checkpoint["extra_info"]["num_heads"]
        test_results["num_layers"] = checkpoint["extra_info"]["num_layers"]
        test_results["lstm_hidden_dim"] = checkpoint["extra_info"]["lstm_hidden_dim"]
        test_results["ffn_hidden_dim"] = [
            tuple(checkpoint["extra_info"]["ffn_hidden_dim"])
            for _ in range(len(test_results.index))
        ]
        test_results["dropout_rate"] = checkpoint["extra_info"]["dropout_rate"]
        test_results["learning_rate"] = checkpoint["extra_info"]["learning_rate"]
        test_results["seed"] = seed
        test_results["loss_function"] = loss
        test_results["gamma"] = gamma
        test_results["k_fold"] = k_fold
        test_results["n_splits"] = n_splits if k_fold else None
        test_results["augmentation_type"] = checkpoint["extra_info"][
            "augmentation_type"
        ]
        test_results["hidden_dim_aug"] = (
            [
                tuple(checkpoint["extra_info"]["hidden_dim_aug"])
                for _ in range(len(test_results.index))
            ]
            if checkpoint["extra_info"]["hidden_dim_aug"] is not None
            else None
        )
        test_results["comb_method"] = checkpoint["extra_info"]["comb_method"]
        test_results["batch_size"] = batch_size
        test_results_df = pd.concat([test_results_df, test_results])

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
