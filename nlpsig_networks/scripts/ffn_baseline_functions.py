from __future__ import annotations

import nlpsig
from nlpsig.classification_utils import DataSplits, Folds
from nlpsig_networks.pytorch_utils import SaveBestModel, training_pytorch, testing_pytorch, set_seed, KFold_pytorch
from nlpsig_networks.ffn import FeedforwardNeuralNetModel
from nlpsig_networks.focal_loss import FocalLoss
import torch
import numpy as np
import pandas as pd
import signatory
from tqdm.auto import tqdm
import os

        
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
    data_split_seed: int = 0,
    k_fold: bool = False,
    n_splits: int = 5,
    verbose_training: bool = True,
    verbose_results: bool = True,
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
    data_split_seed : int, optional
        The seed which is used when splitting, by default 0
    k_fold : bool, optional
        Whether or not to use k-fold validation, by default False
    n_splits : int, optional
        Number of splits to use in k-fold validation, by default 5.
        Ignored if k_fold=False
    verbose_training : bool, optional
        Whether or not to print out training progress, by default True
    verbose_results : bool, optional
        Whether or not to print out results on validation and test, by default True
        
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
    ffn_model = FeedforwardNeuralNetModel(input_dim=x_data.shape[1],
                                          hidden_dim=hidden_dim,
                                          output_dim=output_dim,
                                          dropout_rate=dropout_rate)
    
    if verbose_model:
        print(ffn_model)
    
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
    patience = 10
    
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
        optimizer = torch.optim.Adam(ffn_model.parameters(), lr=learning_rate)
        
        # perform k-fold evaluation which returns a dataframe with columns for the
        # loss, accuracy, f1 (macro) and individual f1-scores for each fold
        # (for both validation and test set)
        results = KFold_pytorch(folds=folds,
                                model=ffn_model,
                                criterion=criterion,
                                optimizer=optimizer,
                                num_epochs=num_epochs,
                                seed=seed,
                                save_best=save_best,
                                early_stopping=early_stopping,
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
            y_train = split_data.get_splits(as_DataLoader=False)[5]
            criterion.set_alpha_from_y(y=y_train)
        elif loss == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("criterion must be either 'focal' or 'cross_entropy'")

        # define optimizer
        optimizer = torch.optim.Adam(ffn_model.parameters(), lr=learning_rate)
        
        # train FFN
        ffn_model = training_pytorch(model=ffn_model,
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
        valid_results = testing_pytorch(model=ffn_model,
                                        test_loader=valid,
                                        criterion=criterion,
                                        verbose=False)
        
        # evaluate on test
        test_results = testing_pytorch(model=ffn_model,
                                       test_loader=test,
                                       criterion=criterion,
                                       verbose=False)
        
        results = pd.DataFrame({"loss": test_results["loss"],
                                "accuracy": test_results["accuracy"], 
                                "f1": test_results["f1"],
                                "f1_scores": [test_results["f1_scores"]],
                                "valid_loss": valid_results["loss"],
                                "valid_accuracy": valid_results["accuracy"], 
                                "valid_f1": valid_results["f1"],
                                "valid_f1_scores": [valid_results["f1_scores"]]})

    if verbose_results:
        with pd.option_context('display.precision', 3):
            print(results)
        
    # remove any models that have been saved
    if os.path.exists(model_output):
        os.remove(model_output)
    
    return ffn_model, results


def ffn_hyperparameter_search(
    num_epochs: int,
    x_data: torch.tensor | np.array,
    y_data: torch.tensor | np.array,
    output_dim: int,
    hidden_dim_sizes : list[list[int]] | list[int],
    dropout_rates: list[float],
    learning_rates: list[float],
    seeds : list[int],
    loss: str,
    gamma: float = 0.0,
    data_split_seed: int = 0,
    k_fold: bool = False,
    n_splits: int = 5,
    validation_metric: str = "f1",
    results_output: str | None = None,
    verbose: bool = True
) -> tuple[pd.DataFrame, float, dict]:
    """
    Performs hyperparameter search for different hidden dimensions,
    dropout rates, learning rates by training and evaluating
    a FFN on various seeds and averaging performance over the seeds.
    
    If k_fold=True, will perform k-fold validation on each seed and
    average over the average performance over the folds, otherwise
    will average over the performance of the FFNs trained using each
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
    hidden_dim_sizes : list[list[int]] | list[int]
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
    data_split_seed : int, optional
        _description_, by default 0
    k_fold : bool, optional
        _description_, by default False
    n_splits : int, optional
        _description_, by default 5
    validation_metric : str, optional
        _description_, by default "f1"
    results_output : str | None, optional
        _description_, by default None
    verbose : bool, optional
        _description_, by default True

    Returns
    -------
    tuple[pd.DataFrame, float, dict]
        _description_
    """
    if validation_metric not in ["loss", "accuracy", "f1"]:
        raise ValueError("validation_metric must be either 'loss', 'accuracy' or 'f1'")
    
    # initialise SaveBestModel class
    model_output = "best_ffn_model.pkl"
    save_best_model = SaveBestModel(metric=validation_metric,
                                    output=model_output,
                                    verbose=verbose)

    # find model parameters that has the best validation
    results_df = pd.DataFrame()
    model_id = 0
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
                    _, results = implement_ffn(num_epochs=num_epochs,
                                               x_data=x_data,
                                               y_data=y_data,
                                               hidden_dim=hidden_dim,
                                               output_dim=output_dim,
                                               dropout_rate=dropout,
                                               learning_rate=lr,
                                               seed=seed,
                                               loss=loss,
                                               gamma=gamma,
                                               data_split_seed=data_split_seed,
                                               k_fold=k_fold,
                                               n_splits=n_splits,
                                               verbose_training=False,
                                               verbose_results=verbose,
                                               verbose_model=verbose_model)
                    
                    # save metric that we want to validate on
                    # taking the mean over the performance on the folds for the seed
                    # if k_fold=False, .mean() just returns the performance for the seed
                    scores.append(results[f"valid_{validation_metric}"].mean())
                    
                    # concatenate to results dataframe
                    results["hidden_dim"] = [hidden_dim for _ in range(len(results.index))]
                    results["dropout_rate"] = dropout
                    results["learning_rate"] = lr
                    results["seed"] = seed
                    results["loss"] = loss
                    results["gamma"] = gamma
                    results["k_fold"] = k_fold
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
                                extra_info={"hidden_dim": hidden_dim,
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
        _, test_results = implement_ffn(num_epochs=num_epochs,
                                        x_data=x_data,
                                        y_data=y_data,
                                        hidden_dim=checkpoint["extra_info"]["hidden_dim"],
                                        output_dim=output_dim,
                                        dropout_rate=checkpoint["extra_info"]["dropout_rate"],
                                        learning_rate=checkpoint["extra_info"]["learning_rate"],
                                        seed=seed,
                                        loss=loss,
                                        gamma=gamma,
                                        data_split_seed=data_split_seed,
                                        k_fold=k_fold,
                                        n_splits=n_splits,
                                        verbose_training=False,
                                        verbose_results=False)
        
        test_results["hidden_dim"] = [checkpoint["extra_info"]["hidden_dim"]
                                      for _ in range(len(results.index))]
        test_results["dropout_rate"] = checkpoint["extra_info"]["dropout_rate"]
        test_results["learning_rate"] = checkpoint["extra_info"]["learning_rate"]
        test_results["seed"] = seed
        test_results["loss"] = loss
        test_results["gamma"] = gamma
        test_results["k_fold"] = k_fold
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


def obtain_mean_history(df: pd.DataFrame,
                        id_column: str,
                        label_column: str,
                        embeddings: np.array,
                        k: int,
                        path_indices: np.array | None = None,
                        concatenate_current: bool = True) -> torch.tensor:
    # use nlpsig to construct the path as a numpy array
    # first define how we construct the path
    path_specifics = {"pad_by": "history",
                      "zero_padding": False,
                      "method": "k_last",
                      "k": k,
                      "time_feature": None,
                      "embeddings": "full",
                      "include_current_embedding": True}
    
    # obtain path by using PrepareData class and .pad method
    paths = nlpsig.PrepareData(df,
                               id_column=id_column,
                               label_column=label_column,
                               embeddings=embeddings)
    path = paths.pad(**path_specifics)
    
    # slice the path in specified way
    if path_indices is not None:
        path = path[path_indices][:,:,:-2]

    # remove last two columns (which contains the id and the label)
    path = path[:,:,:-2]
    
    # average in the first dimension to pool embeddings in the path
    path = path.mean(1).astype("float")
    
    # concatenate with current embedding (and convert to torch tensor)
    if concatenate_current:
        path =  torch.cat([torch.from_numpy(path),
                           torch.from_numpy(embeddings[path_indices])],
                          dim=1).float()
    else:
        path = torch.from_numpy(path).float()

    return path


def obtain_signatures_history(method: str,
                              dimension: int,
                              sig_depth: int,
                              log_signature: bool,
                              df: pd.DataFrame,
                              id_column: str,
                              label_column: str,
                              embeddings: np.array,
                              k: int,
                              seed: int = 42,
                              path_indices: np.array | None = None,
                              concatenate_current: bool = True) -> torch.tensor:
    # use nlpsig to construct the path as a numpy array
    # first define how we construct the path
    path_specifics = {"pad_by": "history",
                      "zero_padding": True,
                      "method": "k_last",
                      "k": k,
                      "time_feature": None,
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
    path = paths.pad(**path_specifics)
    
    # slice the path in specified way
    if path_indices is not None:
        path = path[path_indices][:,:,:-2]

    # remove last two columns (which contains the id and the label)
    path = path[:,:,:-2].astype("float")
    
    # convert to torch tensor to compute signature using signatory
    path = torch.from_numpy(path).float()
    if log_signature:
        sig = signatory.signature(path, sig_depth).float()
    else:
        sig = signatory.logsignature(path, sig_depth).float()
    
    # concatenate with current embedding
    if concatenate_current:
        sig = torch.cat([sig, torch.from_numpy(embeddings[path_indices])],
                        dim=1)

    return sig


def histories_baseline_hyperparameter_search(
    num_epochs: int,
    df: pd.DataFrame,
    id_column: str,
    label_column: str,
    embeddings: np.array,
    y_data: np.array,
    output_dim: int,
    window_sizes: list[int],
    hidden_dim_sizes : list[list[int]] | list[int],
    dropout_rates: list[float],
    learning_rates: list[float],
    use_signatures: bool,    
    seeds : list[int],
    loss: str,
    gamma: float = 0.0,
    log_signature: bool = False,
    dim_reduce_methods: list[str] | None = None,
    dimension_and_sig_depths: list[tuple[int, int]] | None = None,
    path_indices: np.array | None = None,
    data_split_seed: int = 0,
    k_fold: bool = False,
    n_splits: int = 5,
    validation_metric: str = "f1",
    results_output: str | None = None,
    verbose: bool = True
) -> tuple[pd.DataFrame, float, dict]:
    if use_signatures:
        if dim_reduce_methods is None:
            msg = "if use_signatures=True, must pass in the methods for dimension reduction"
            raise ValueError(msg)
        if dimension_and_sig_depths is None:
            msg = "if use_signatures=True, must pass in the dimensions and signature depths"
            raise ValueError(msg)
        
    # initialise SaveBestModel class
    model_output = "best_ffn_history_model.pkl"
    best_model = SaveBestModel(metric=validation_metric,
                               output=model_output,
                               verbose=verbose)
    
    results_df = pd.DataFrame()
    model_id = 0
    for k in tqdm(window_sizes):
        if verbose:
            print("\n" + "-" * 50)
            print(f"k: {k}")
        if use_signatures:
            for dimension, sig_depth in tqdm(dimension_and_sig_depths):
                for method in tqdm(dim_reduce_methods):
                    if verbose:
                        print("\n" + "#" * 50)
                        print(f"dimension: {dimension} | "
                            f"sig_depth: {sig_depth} | "
                            f"method: {method}")
                    
                    # obtain the ffn input by dimension reduction and computing signatures
                    x_data = obtain_signatures_history(method=method,
                                                       dimension=dimension,
                                                       sig_depth=sig_depth,
                                                       log_signature=log_signature,
                                                       df=df,
                                                       id_column=id_column,
                                                       label_column=label_column,
                                                       embeddings=embeddings,
                                                       k=k,
                                                       path_indices=path_indices,
                                                       concatenate_current=True)

                    # perform hyperparameter search for FFN
                    results, best_valid_metric, FFN_info = ffn_hyperparameter_search(
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
                        data_split_seed=data_split_seed,
                        k_fold=k_fold,
                        n_splits=n_splits,
                        validation_metric=validation_metric,
                        results_output=None,
                        verbose=False
                    )

                    # concatenate to results dataframe
                    results["k"] = k
                    results["input_dim"] = x_data.shape[1]
                    results["dimension"] = dimension
                    results["sig_depth"] = sig_depth
                    results["method"] = method
                    results["log_signature"] = log_signature
                    results["model_id"] = [float(f"{model_id}.{id}") for id in results["model_id"]]
                    results_df = pd.concat([results_df, results])
                    
                    best_model(current_valid_metric=best_valid_metric,
                               extra_info={"k": k,
                                           "input_dim": x_data.shape[1],
                                           "dimension": dimension,
                                           "sig_depth": sig_depth,
                                           "method": method,
                                           "log_signature": log_signature,
                                           **FFN_info})
                    
                    model_id += 1
        else:
            # obtain the ffn input by averaging over history
            x_data = obtain_mean_history(df=df,
                                         id_column=id_column,
                                         label_column=label_column,
                                         embeddings=embeddings,
                                         k=k,
                                         path_indices=path_indices,
                                         concatenate_current=True)
            
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
                data_split_seed=data_split_seed,
                k_fold=k_fold,
                n_splits=n_splits,
                validation_metric=validation_metric,
                results_output=None,
                verbose=False
            )

            # concatenate to results dataframe
            results["k"] = k
            results["input_dim"] = x_data.shape[1]
            results["model_id"] = [float(f"{model_id}.{id}") for id in results["model_id"]]
            results_df = pd.concat([results_df, results])
            
            best_model(current_valid_metric=best_valid_metric,
                        extra_info={"k": k,
                                    "input_dim": x_data.shape[1],
                                    **FFN_info})
            
            model_id += 1

    checkpoint = torch.load(f=model_output)
    if verbose:
        print("*" * 50)
        print("The best model had the following parameters:")
        print(checkpoint["extra_info"])
    
    if use_signatures:
        # obtain the ffn input by dimension reduction and computing signatures
        x_data = obtain_signatures_history(method=checkpoint["extra_info"]["method"],
                                           dimension=checkpoint["extra_info"]["dimension"],
                                           sig_depth=checkpoint["extra_info"]["sig_depth"],
                                           log_signature=checkpoint["extra_info"]["log_signature"],
                                           df=df,
                                           id_column=id_column,
                                           label_column=label_column,
                                           embeddings=embeddings,
                                           k=checkpoint["extra_info"]["k"],
                                           path_indices=path_indices,
                                           concatenate_current=True)
    else:
        # obtain the ffn input by averaging over history
        x_data = obtain_mean_history(df=df,
                                     id_column=id_column,
                                     label_column=label_column,
                                     embeddings=embeddings,
                                     k=checkpoint["extra_info"]["k"],
                                     path_indices=path_indices,
                                     concatenate_current=True)
    
    test_scores = []
    test_results_df = pd.DataFrame()
    for seed in seeds:
        _, test_results = implement_ffn(num_epochs=num_epochs,
                                        x_data=x_data,
                                        y_data=y_data,
                                        output_dim=output_dim,
                                        hidden_dim=checkpoint["extra_info"]["hidden_dim"],
                                        dropout_rate=checkpoint["extra_info"]["dropout_rate"],
                                        learning_rate=checkpoint["extra_info"]["learning_rate"],
                                        seed=seed,
                                        loss=loss,
                                        gamma=gamma,
                                        data_split_seed=data_split_seed,
                                        k_fold=k_fold,
                                        n_splits=n_splits,
                                        verbose_training=False,
                                        verbose_results=False)
        
        test_results["hidden_dim"] = [checkpoint["extra_info"]["hidden_dim"]
                                      for _ in range(len(test_results.index))]
        test_results["dropout_rate"] = checkpoint["extra_info"]["dropout_rate"]
        test_results["learning_rate"] = checkpoint["extra_info"]["learning_rate"]
        test_results["seed"] = seed
        test_results["loss"] = loss
        test_results["gamma"] = gamma
        test_results["k_fold"] = k_fold
        test_results["k"] = checkpoint["extra_info"]["k"]
        test_results["input_dim"] = checkpoint["extra_info"]["input_dim"]
        if use_signatures:
            test_results["dimension"] = checkpoint["extra_info"]["dimension"]
            test_results["sig_depth"] = checkpoint["extra_info"]["sig_depth"]
            test_results["method"] = checkpoint["extra_info"]["method"]
            results["log_signature"] = checkpoint["extra_info"]["log_signature"]
        test_results_df = pd.concat([test_results_df, test_results])
        
        # save metric that we want to validate on
        # taking the mean over the performance on the folds for the seed
        # if k_fold=False, .mean() just returns the performance for the seed
        test_scores.append(test_results[validation_metric].mean())
        
    test_scores_mean = sum(test_scores)/len(test_scores)
    
    if verbose:
        print(f"best validation score: {best_model.best_valid_metric}")
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
        
    return results_df, test_results_df, best_model.best_valid_metric, checkpoint["extra_info"]