from __future__ import annotations

import nlpsig
from nlpsig.classification_utils import DataSplits, Folds
from nlpsig_networks.pytorch_utils import SaveBestModel, training_pytorch, testing_pytorch, set_seed, KFold_pytorch
from nlpsig_networks.swnu_network import SWNUNetwork
from nlpsig_networks.focal_loss import FocalLoss
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os


def obtain_SWNUNetwork_input(
    method: str,
    dimension: int,
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    embeddings: np.array,
    k: int,
    time_feature: list[str] | str | None = None,
    standardise_method: list[str] | str | None = None,
    add_time_in_path: bool = False,
    seed: int = 42,
    path_indices : list | np.array | None = None
) -> tuple[torch.tensor, int]:
    # use nlpsig to construct the path as a numpy array
    # first define how we construct the path
    path_specifics = {"pad_by": "history",
                      "zero_padding": True,
                      "method": "k_last",
                      "k": k,
                      "time_feature": time_feature,
                      "standardise_method": standardise_method,
                      "embeddings": "dim_reduced",
                      "include_current_embedding": True}
    
    # first perform dimension reduction on embeddings
    if dimension == embeddings.shape[1]:
        # no need to perform dimensionality reduction
        embeddings_reduced = embeddings
    else:
        reduction = nlpsig.DimReduce(method=method,
                                     n_components=dimension)
        embeddings_reduced = reduction.fit_transform(embeddings,
                                                     random_state=seed)
    
    # obtain path by using PrepareData class and .pad method
    paths = nlpsig.PrepareData(df,
                               id_column=id_column,
                               label_column=label_column,
                               embeddings=embeddings,
                               embeddings_reduced=embeddings_reduced)
    paths.pad(**path_specifics)
    
    # slice the path in specified way
    if path_indices is not None:
        paths.array_padded = paths.array_padded[path_indices]
        paths.embeddings = paths.embeddings[path_indices]
        paths.embeddings_reduced = paths.embeddings_reduced[path_indices]
    
    return paths.get_torch_path_for_SWNUNetwork(
        include_time_features_in_path=add_time_in_path,
        include_time_features_in_input=True,
        include_embedding_in_input=True,
        reduced_embeddings=False
    )
    
def implement_swnu_network(
    num_epochs: int,
    x_data: torch.tensor | np.array,
    y_data: torch.tensor | np.array,
    input_channels: int,
    output_channels: int,
    num_time_features: int,
    embedding_dim: int,
    log_signature: bool,
    sig_depth: int,
    lstm_hidden_dim: list[int] | int,
    ffn_hidden_dim: list[int] | int,
    output_dim: int,
    BiLSTM: bool,
    dropout_rate: float,
    learning_rate: float,
    seed: int,
    loss: str,
    gamma: float = 0.0,
    augmentation_type: str = "Conv1d",
    hidden_dim_aug: list[int] | int | None = None,
    comb_method: str = "concatenation",
    data_split_seed: int = 0,
    k_fold: bool = False,
    n_splits: int = 5,
    patience: int = 10,
    verbose_training: bool = True,
    verbose_results: bool = True,
    verbose_model: bool = False,
) -> tuple[SWNUNetwork, pd.DataFrame]:
    # set seed
    set_seed(seed)
    
    # initialise SWNUNetwork
    SWNUNetwork_args = {
        "input_channels": input_channels,
        "output_channels": output_channels,
        "num_time_features": num_time_features,
        "embedding_dim": embedding_dim,
        "log_signature": log_signature,
        "sig_depth": sig_depth,
        "hidden_dim_swnu": lstm_hidden_dim,
        "hidden_dim_ffn": ffn_hidden_dim,
        "output_dim": output_dim,
        "dropout_rate": dropout_rate,
        "augmentation_type": augmentation_type,
        "hidden_dim_aug": hidden_dim_aug,
        "BiLSTM": BiLSTM,
        "comb_method": comb_method
    }
    swnu_network_model = SWNUNetwork(**SWNUNetwork_args)
    
    if verbose_model:
        print(swnu_network_model)
    
    # convert data to torch tensors
    if not isinstance(x_data, torch.Tensor):
        x_data = torch.tensor(x_data)
    if not isinstance(y_data, torch.Tensor):
        y_data = torch.tensor(y_data)
    x_data = x_data.float()
    
    # set some variables for training
    save_best = True
    early_stopping = True
    model_output = "best_model.pkl"
    validation_metric = "f1"
    weight_decay_adam = 0.0001
    
    if k_fold:
        # perform KFold evaluation and return the performance on validation and test sets
        # split dataset
        folds = Folds(x_data=x_data,
                      y_data=y_data,
                      n_splits=n_splits,
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
        optimizer = torch.optim.Adam(swnu_network_model.parameters(), lr=learning_rate, weight_decay= weight_decay_adam)
        
        # perform k-fold evaluation which returns a dataframe with columns for the
        # loss, accuracy, f1 (macro) and individual f1-scores for each fold
        # (for both validation and test set)
        results = KFold_pytorch(folds=folds,
                                model=swnu_network_model,
                                criterion=criterion,
                                optimizer=optimizer,
                                num_epochs=num_epochs,
                                seed=seed,
                                save_best=save_best,
                                early_stopping=early_stopping,
                                validation_metric=validation_metric,
                                patience=patience,
                                verbose=verbose_training)
    else:
        # split dataset
        split_data = DataSplits(x_data=x_data,
                                y_data=y_data,
                                train_size=0.8,
                                valid_size=0.2,
                                shuffle=True,
                                random_state=data_split_seed)
        train, valid, test = split_data.get_splits(as_DataLoader=True)

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
        optimizer = torch.optim.Adam(swnu_network_model.parameters(), lr=learning_rate, weight_decay= weight_decay_adam)
        
        # train FFN
        swnu_network_model = training_pytorch(model=swnu_network_model,
                                      train_loader=train,
                                      criterion=criterion,
                                      optimizer=optimizer,
                                      num_epochs=num_epochs,
                                      valid_loader=valid,
                                      seed=seed,
                                      save_best=save_best,
                                      output=model_output,
                                      early_stopping=early_stopping,
                                      validation_metric=validation_metric,
                                      patience=patience,
                                      verbose=verbose_training)
        
        # evaluate on validation
        test_results = testing_pytorch(model=swnu_network_model,
                                       test_loader=test,
                                       criterion=criterion,
                                       verbose=False)
        
        # evaluate on test
        valid_results = testing_pytorch(model=swnu_network_model,
                                        test_loader=valid,
                                        criterion=criterion)
        
        results = pd.DataFrame({"loss": test_results["loss"],
                                "accuracy": test_results["accuracy"], 
                                "f1": test_results["f1"],
                                "f1_scores": test_results["f1_scores"],
                                "valid_loss": valid_results["loss"],
                                "valid_accuracy": valid_results["accuracy"], 
                                "valid_f1": valid_results["f1"],
                                "valid_f1_scores": valid_results["f1_scores"]})

    if verbose_results:
        with pd.option_context('display.precision', 3):
            print(results)
            
    # remove any models that have been saved
    if os.path.exists(model_output):
        os.remove(model_output)
        
    return swnu_network_model, results


def swnu_network_hyperparameter_search(
    num_epochs: int,
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    embeddings: np.array,
    y_data: np.array,
    embedding_dim: int,
    output_dim: int,
    history_lengths: list[int],
    dim_reduce_methods: list[str],
    dimensions: list[int],
    sig_depths: list[int],
    log_signature: bool,
    conv_output_channels: list[int],
    swnu_hidden_dim_sizes: list[int] | list[list[int]],
    ffn_hidden_dim_sizes: list[int] | list[list[int]],
    dropout_rates: list[float],
    learning_rates: list[float],
    BiLSTM,
    seeds : list[int],
    loss: str,
    gamma: float = 0.0,
    time_feature: list[str] | str | None = None,
    standardise_method: list[str] | str | None = None,
    add_time_in_path: bool = False,
    augmentation_type: str = "Conv1d",
    hidden_dim_aug: list[int] | int | None = None,
    comb_method: str = "concatenation",
    path_indices : list | np.array | None = None,
    data_split_seed: int = 0,
    k_fold: bool = False,
    n_splits: int = 5,
    patience: int = 10,
    validation_metric: str = "f1",
    results_output: str | None = None,
    verbose: bool = True
):
    if validation_metric not in ["loss", "accuracy", "f1"]:
        raise ValueError("validation_metric must be either 'loss', 'accuracy' or 'f1'")
    
    # initialise SaveBestModel class
    model_output = "best_swnu_network_model.pkl",
    save_best_model = SaveBestModel(metric=validation_metric,
                                    output=model_output,
                                    verbose=verbose)
    
    results_df = pd.DataFrame()
    model_id = 0

    if isinstance(time_feature, str):
        time_feature = [time_feature]
    
    for k in tqdm(history_lengths):
        if verbose:
            print("\n" + "-" * 50)
            print(f"k: {k}")
        for dimension in tqdm(dimensions):
            for method in tqdm(dim_reduce_methods):
                print("\n" + "#" * 50)
                print(f"dimension: {dimension} | "
                      f"method: {method}")
                x_data, input_channels = obtain_SWNUNetwork_input(
                    method=method,
                    dimension=dimension,
                    df=df,
                    id_column=id_column,
                    label_column=label_column,
                    embeddings=embeddings,
                    k=k,
                    time_feature=time_feature,
                    standardise_method=standardise_method,
                    add_time_in_path= add_time_in_path,
                    path_indices=path_indices
                )
        
                for lstm_hidden_dim in tqdm(swnu_hidden_dim_sizes):
                    for ffn_hidden_dim in tqdm(ffn_hidden_dim_sizes):
                        for sig_depth in sig_depths:
                            for output_channels in tqdm(conv_output_channels):
                                for dropout in tqdm(dropout_rates):
                                    for lr in tqdm(learning_rates):
                                        if verbose:
                                            print("\n" + "!" * 50)
                                            print(f"lstm_hidden_dim: {lstm_hidden_dim} | "
                                                  f"ffn_hidden_dim: {ffn_hidden_dim} | "
                                                  f"sig_depth: {sig_depth} | "
                                                  f"output_channels: {output_channels} | "
                                                  f"dropout: {dropout} | "
                                                  f"learning_rate: {lr}")
                                        scores = []
                                        verbose_model = verbose
                                        for seed in seeds:
                                            _, results = implement_swnu_network(
                                                num_epochs=num_epochs,
                                                x_data=x_data,
                                                y_data=y_data,
                                                input_channels=input_channels,
                                                output_channels=output_channels,
                                                num_time_features=len(time_feature),
                                                embedding_dim=embedding_dim,
                                                log_signature=log_signature,
                                                sig_depth=sig_depth,
                                                lstm_hidden_dim=lstm_hidden_dim,
                                                ffn_hidden_dim=ffn_hidden_dim,
                                                output_dim=output_dim,
                                                BiLSTM=BiLSTM,
                                                dropout_rate=dropout,
                                                learning_rate=lr,
                                                seed=seed,
                                                loss=loss,
                                                gamma=gamma,
                                                augmentation_type=augmentation_type,
                                                hidden_dim_aug=hidden_dim_aug,
                                                comb_method=comb_method,
                                                data_split_seed=data_split_seed,
                                                k_fold=k_fold,
                                                n_splits=n_splits,
                                                patience=patience,
                                                verbose_training=False,
                                                verbose_results=verbose,
                                                verbose_model=verbose_model
                                            )
                                            # save metric that we want to validate on
                                            # taking the mean over the performance on the folds for the seed
                                            # if k_fold=False, .mean() just returns the performance for the seed
                                            scores.append(results[f"valid_{validation_metric}"].mean())
                                            
                                            # concatenate to results dataframe
                                            results["k"] = k
                                            results["dimensions"] = dimension
                                            results["sig_depth"] = sig_depth
                                            results["method"] = method
                                            results["input_channels"] = input_channels
                                            results["output_channels"] = output_channels
                                            results["num_time_features"] = len(time_feature)
                                            results["embedding_dim"] = embedding_dim
                                            results["log_signature"] = log_signature
                                            results["lstm_hidden_dim"] = [lstm_hidden_dim for _ in range(len(results.index))]
                                            results["ffn_hidden_dim"] = [ffn_hidden_dim for _ in range(len(results.index))]
                                            results["dropout_rate"] = dropout
                                            results["learning_rate"] = lr
                                            results["seed"] = seed
                                            results["BiLSTM"] = BiLSTM
                                            results["loss"] = loss
                                            results["gamma"] = gamma
                                            results["k_fold"] = k_fold
                                            results["augmentation_type"] = augmentation_type
                                            results["hidden_dim_aug"] = [hidden_dim_aug for _ in range(len(results.index))]
                                            results["comb_method"] = comb_method
                                            results["model_id"] = model_id
                                            results_df = pd.concat([results_df, results])
                                            
                                            # don't continue printing out the model
                                            verbose_model = False

                                        model_id += 1
                                        scores_mean = sum(scores)/len(scores)
                                        
                                        print(f"- average{' (kfold)' if k_fold else ''} "
                                              f"(validation) metric score: {scores_mean}")
                                        print(f"scores for the different seeds: {scores}")
                                        # save best model according to averaged metric over the different seeds
                                        save_best_model(current_valid_metric=scores_mean,
                                                        extra_info={
                                                            "k": k,
                                                            "dimensions": dimension,
                                                            "sig_depth": sig_depth,
                                                            "method": method,
                                                            "input_channels": input_channels,
                                                            "output_channels": output_channels,
                                                            "num_time_features": len(time_feature),
                                                            "embedding_dim": embedding_dim,
                                                            "log_signature": log_signature,
                                                            "lstm_hidden_dim": lstm_hidden_dim,
                                                            "ffn_hidden_dim": ffn_hidden_dim,
                                                            "dropout_rate": dropout,
                                                            "learning_rate": lr,
                                                            "BiLSTM": BiLSTM,
                                                            "loss": loss,
                                                            "gamma": gamma,
                                                            "augmentation_type": augmentation_type,
                                                            "hidden_dim_aug": hidden_dim_aug,
                                                            "comb_method": comb_method
                                                        })

    checkpoint = torch.load(f=model_output)
    if verbose:
        print("*" * 50)
        print("The best model had the following parameters:")
        print(checkpoint["extra_info"])

    x_data, input_channels = obtain_SWNUNetwork_input(method=checkpoint["extra_info"]["method"],
                                               dimension=checkpoint["extra_info"]["k"],
                                               df=df,
                                               id_column=id_column,
                                               label_column=label_column,
                                               embeddings=embeddings,
                                               k=checkpoint["extra_info"]["k"],
                                               path_indices=path_indices)

    test_scores = []
    test_results_df = pd.DataFrame()
    for seed in seeds:
        _, test_results = implement_swnu_network(
            num_epochs=num_epochs,
            x_data=x_data,
            y_data=y_data,
            sig_depth=checkpoint["extra_info"]["sig_depth"],
            input_channels=checkpoint["extra_info"]["input_channels"],
            output_channels=checkpoint["extra_info"]["output_channels"],
            num_time_features=len(time_feature),
            embedding_dim=embedding_dim,
            log_signature=checkpoint["extra_info"]["log_signature"],
            output_dim=output_dim,
            lstm_hidden_dim=checkpoint["extra_info"]["lstm_hidden_dim"],
            ffn_hidden_dim=checkpoint["extra_info"]["ffn_hidden_dim"],
            BiLSTM=checkpoint["extra_info"]["BiLSTM"],
            dropout_rate=checkpoint["extra_info"]["dropout_rate"],
            learning_rate=checkpoint["extra_info"]["learning_rate"],
            seed=seed,
            loss=checkpoint["extra_info"]["loss"],
            gamma=checkpoint["extra_info"]["gamma"],
            augmentation_type=checkpoint["extra_info"]["augmentation_type"],
            hidden_dim_aug = checkpoint["extra_info"]["hidden_dim_aug"],
            comb_method=checkpoint["extra_info"]["comb_method"],
            data_split_seed=data_split_seed,
            k_fold=k_fold,
            n_splits=n_splits,
            patience=patience,
            verbose_training=False,
            verbose_results=False,
            verbose_model=False
        )

        # save metric that we want to validate on
        # taking the mean over the performance on the folds for the seed
        # if k_fold=False, .mean() just returns the performance for the seed
        test_scores.append(test_results[validation_metric].mean())
        
        # concatenate to results dataframe
        test_results["k"] = checkpoint["extra_info"]["k"]
        test_results["dimensions"] = checkpoint["extra_info"]["dimensions"]
        test_results["sig_depth"] = checkpoint["extra_info"]["sig_depth"]
        test_results["method"] = checkpoint["extra_info"]["method"]
        test_results["input_channels"] = checkpoint["extra_info"]["input_channels"]
        test_results["output_channels"] = checkpoint["extra_info"]["output_channels"]
        test_results["num_time_features"] = len(time_feature)
        test_results["embedding_dim"] = embedding_dim
        test_results["log_signature"] = checkpoint["extra_info"]["log_signature"]
        test_results["lstm_hidden_dim"] = [checkpoint["extra_info"]["lstm_hidden_dim"]
                                           for _ in range(len(test_results.index))]
        test_results["ffn_hidden_dim"] = [checkpoint["extra_info"]["ffn_hidden_dim"]
                                          for _ in range(len(test_results.index))]
        test_results["dropout_rate"] = checkpoint["extra_info"]["dropout_rate"]
        test_results["learning_rate"] = checkpoint["extra_info"]["learning_rate"]
        test_results["seed"] = seed
        test_results["BiLSTM"] = checkpoint["extra_info"]["BiLSTM"]
        test_results["loss"] = checkpoint["extra_info"]["loss"]
        test_results["gamma"] = checkpoint["extra_info"]["gamma"]
        test_results["k_fold"] = k_fold
        test_results["augmentation_type"] = checkpoint["extra_info"]["augmentation_type"]
        test_results["hidden_dim_aug"] = checkpoint["extra_info"]["hidden_dim_aug"]
        test_results["comb_method"] = checkpoint["extra_info"]["comb_method"]
        test_results_df = pd.concat([test_results_df, test_results])
        
    test_scores_mean = sum(test_scores)/len(test_scores)
    if verbose:
        print(f"best validation score: {save_best_model.best_valid_metric}")
        print(f"- Best model: average (test) metric score: {test_scores_mean}")
        print(f"scores for the different seeds: {test_scores}")
        
    if results_output is not None:
        print("saving results dataframe to CSV for this "
            f"hyperparameter search in {results_output}")
        results_df.to_csv(results_output)
    
    # remove any models that have been saved
    if os.path.exists(model_output):
        os.remove(model_output)
    
    return results_df, test_results_df, save_best_model.best_valid_metric, checkpoint["extra_info"]