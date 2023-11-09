from __future__ import annotations

import os
import shutil
from typing import Iterable

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets.arrow_dataset import Dataset
from nlpsig import TextEncoder
from nlpsig.classification_utils import DataSplits, Folds
from sklearn import metrics
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)
from tqdm.auto import tqdm

from nlpsig_networks.focal_loss import FocalLoss
from nlpsig_networks.pytorch_utils import SaveBestModel, _get_timestamp, set_seed


def testing_transformer(
    trainer: Trainer,
    test_dataset: Dataset,
    verbose: bool = False,
) -> dict[str, float | list[float]]:
    """
    Function to evaluate a transformer model by computing the accuracy
    and F1 score.
    """
    predictions = trainer.predict(test_dataset)
    predicted = np.argmax(predictions.predictions, axis=-1)

    # convert to torch tensor
    predicted = torch.tensor(predicted)
    labels = torch.tensor(predictions.label_ids)

    # compute accuracy
    accuracy = ((predicted == labels).sum() / len(labels)).item()

    # compute F1 scores
    f1_scores = metrics.f1_score(labels, predicted, average=None, zero_division=0.0)
    # compute macro F1 score
    f1 = sum(f1_scores) / len(f1_scores)

    # compute precision scores
    precision_scores = metrics.precision_score(
        labels, predicted, average=None, zero_division=0.0
    )
    # compute macro precision score
    precision = sum(precision_scores) / len(precision_scores)

    # compute recall scores
    recall_scores = metrics.recall_score(
        labels, predicted, average=None, zero_division=0.0
    )
    # compute macro recall score
    recall = sum(recall_scores) / len(recall_scores)

    if verbose:
        # print evaluation metrics
        print(f"Accuracy on dataset of size {len(labels)}: " f"{100 * accuracy} %.")
        print(f"- f1: {f1_scores}")
        print(f"- f1 (macro): {f1}")
        print(f"- precision (macro): {precision}")
        print(f"- recall (macro): {recall}")

    return {
        "predicted": predicted,
        "labels": labels,
        "accuracy": accuracy,
        "f1": f1,
        "f1_scores": [f1_scores],
        "precision": precision,
        "precision_scores": [precision_scores],
        "recall": recall,
        "recall_scores": [recall_scores],
    }


def _fine_tune_transformer_for_data_split(
    num_epochs: int,
    pretrained_model_name: str,
    df: pd.DataFrame,
    feature_name: str,
    label_column: str,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    output_dim: int,
    learning_rate: float,
    seed: int,
    loss: str,
    gamma: float = 0.0,
    device: str | None = None,
    batch_size: int = 64,
    path_indices: list | np.array | None = None,
    split_indices: tuple[Iterable[int] | None] | None = None,
    save_model: bool = False,
    output_dir: str | None = None,
    verbose: bool = False,
) -> dict[str, float | list[float]]:
    """
    Function to fine-tune and evalaute a model for a given
    data_split (via split_indices)
    """
    # set seed
    set_seed(seed)

    if output_dir is None:
        output_dir = f"fine_tuned_{pretrained_model_name}"
    if not isinstance(output_dir, str):
        raise TypeError("output_dir must be either a string or None")

    if path_indices is not None:
        df = df.iloc[path_indices].reset_index(drop=True)

    # define loss
    if loss == "focal":
        y_data = df[label_column]
        criterion = FocalLoss(gamma=gamma)
        y_train = torch.tensor(y_data.apply(lambda x: label_to_id[str(x)]).values)
        criterion.set_alpha_from_y(y=y_train)
    elif loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("loss must be either 'focal' or 'cross_entropy'")

    # create column named "label_as_id" which are the corresponding IDs
    df["label_as_id"] = df[label_column].apply(lambda x: label_to_id[str(x)])
    # create labels column
    # NOTE: this is needed for using a custom loss in TextEncoder class
    df["labels"] = df["label_as_id"]

    # initialise model, tokenizer and data_collator
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=output_dim,
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # set model to device is passed
    if isinstance(device, str):
        model.to(device)

    # set tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # initialise TextEncoder object from nlpsig
    # use this to fine-tune the transformer to classification task
    text_encoder = TextEncoder(
        df=df,
        feature_name=feature_name,
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        verbose=False,
    )

    # tokenize the text in df[feature_name]
    text_encoder.tokenize_text()

    # split the dataset using the indices which are passed in
    text_encoder.split_dataset(indices=split_indices)

    # set up training arguments
    text_encoder.set_up_training_args(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        disable_tqdm=False,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="Validation F1",
        seed=seed,
    )

    # set up trainer
    def _compute_metrics(eval_pred):
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        predictions = np.argmax(eval_pred.predictions, axis=1)
        accuracy = accuracy.compute(
            predictions=predictions, references=eval_pred.label_ids
        )["accuracy"]
        f1 = f1.compute(
            predictions=predictions, references=eval_pred.label_ids, average="macro"
        )["f1"]
        return {"Validation Accuracy": accuracy, "Validation F1": f1}

    text_encoder.set_up_trainer(
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        custom_loss=criterion.forward,
    )

    # train model
    text_encoder.fit_transformer_with_trainer_api()

    # evaluate on validation set
    validation_performance = testing_transformer(
        trainer=text_encoder.trainer,
        test_dataset=text_encoder.dataset_split["validation"],
        verbose=verbose,
    )

    # evaluate on test set
    test_performance = testing_transformer(
        trainer=text_encoder.trainer,
        test_dataset=text_encoder.dataset_split["test"],
        verbose=verbose,
    )

    if save_model:
        text_encoder.trainer.save_model(output_dir)
    else:
        # if do not request to save the model,
        # make sure to delete any folders created
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

    return {
        "predicted": test_performance["predicted"],
        "labels": test_performance["labels"],
        "accuracy": test_performance["accuracy"],
        "f1": test_performance["f1"],
        "f1_scores": [test_performance["f1_scores"]],
        "precision": test_performance["precision"],
        "precision_scores": [test_performance["precision_scores"]],
        "recall": test_performance["recall"],
        "recall_scores": [test_performance["recall_scores"]],
        "valid_predicted": validation_performance["predicted"],
        "valid_labels": validation_performance["labels"],
        "valid_accuracy": validation_performance["accuracy"],
        "valid_f1": validation_performance["f1"],
        "valid_f1_scores": [validation_performance["f1_scores"]],
        "valid_precision": validation_performance["precision"],
        "valid_precision_scores": [validation_performance["precision_scores"]],
        "valid_recall": validation_performance["recall"],
        "valid_recall_scores": [validation_performance["recall_scores"]],
    }


def fine_tune_transformer_for_classification(
    num_epochs: int,
    pretrained_model_name: str,
    df: pd.DataFrame,
    feature_name: str,
    label_column: str,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    output_dim: int,
    learning_rate: float,
    seed: int,
    loss: str,
    gamma: float = 0.0,
    device: str | None = None,
    batch_size: int = 64,
    path_indices: list | np.array | None = None,
    data_split_seed: int = 0,
    split_ids: torch.Tensor | None = None,
    split_indices: tuple[Iterable[int] | None]
    | tuple[tuple[Iterable[int] | None]]
    | None = None,
    k_fold: bool = False,
    n_splits: int = 5,
    return_metric_for_each_fold: bool = False,
    verbose: bool = False,
):
    """
    Function to fine-tune and evaluate a model by either using k_fold
    evaluation or a standard train/valid/test split.
    """
    # set seed
    set_seed(seed)

    # create dummy dataset for passing into Folds and DataSplit
    datasize = (
        len(df.index) if path_indices is None else len(df.iloc[path_indices].index)
    )
    dummy_data = torch.ones(datasize)

    if k_fold:
        # perform KFold evaluation and return the performance
        # on validation and test sets
        # first split dataset
        folds = Folds(
            x_data=dummy_data,
            y_data=dummy_data,
            groups=split_ids,
            n_splits=n_splits,
            indices=split_indices,
            shuffle=True,
            random_state=data_split_seed,
        )

        # create lists to record the test metrics for each fold
        accuracy = []
        f1 = []
        f1_scores = []
        precision = []
        precision_scores = []
        recall = []
        recall_scores = []

        # create lists to record the metrics evaluated on the
        # validation sets for each fold
        valid_accuracy = []
        valid_f1 = []
        valid_f1_scores = []
        valid_precision = []
        valid_precision_scores = []
        valid_recall = []
        valid_recall_scores = []

        labels = torch.empty((0))
        predicted = torch.empty((0))
        valid_labels = torch.empty((0))
        valid_predicted = torch.empty((0))
        for k in range(n_splits):
            # compute how well the model performs on this fold
            results_for_fold = _fine_tune_transformer_for_data_split(
                num_epochs=num_epochs,
                pretrained_model_name=pretrained_model_name,
                df=df,
                feature_name=feature_name,
                label_column=label_column,
                label_to_id=label_to_id,
                id_to_label=id_to_label,
                output_dim=output_dim,
                learning_rate=learning_rate,
                seed=seed,
                loss=loss,
                gamma=gamma,
                device=device,
                batch_size=batch_size,
                path_indices=path_indices,
                split_indices=folds.fold_indices[k],
                save_model=False,
                verbose=verbose,
            )

            # store the true labels and predicted labels for this fold
            labels = torch.cat([labels, results_for_fold["labels"]])
            predicted = torch.cat([predicted, results_for_fold["predicted"]])

            # store the metrics for this fold
            accuracy.append(results_for_fold["accuracy"])
            f1.append(results_for_fold["f1"])
            f1_scores.append(results_for_fold["f1_scores"])
            precision.append(results_for_fold["precision"])
            precision_scores.append(results_for_fold["precision_scores"])
            recall.append(results_for_fold["recall"])
            recall_scores.append(results_for_fold["recall_scores"])

            # store the true labels and predicted labels for
            # this fold on the validation set
            valid_labels = torch.cat([valid_labels, results_for_fold["valid_labels"]])
            valid_predicted = torch.cat(
                [valid_predicted, results_for_fold["valid_predicted"]]
            )

            # store the metrics for this fold
            valid_accuracy.append(results_for_fold["valid_accuracy"])
            valid_f1.append(results_for_fold["valid_f1"])
            valid_f1_scores.append(results_for_fold["valid_f1_scores"])
            valid_precision.append(results_for_fold["valid_precision"])
            valid_precision_scores.append(results_for_fold["valid_precision_scores"])
            valid_recall.append(results_for_fold["valid_recall"])
            valid_recall_scores.append(results_for_fold["valid_recall_scores"])

        if not return_metric_for_each_fold:
            # compute how well the model performed on the test sets together
            # compute accuracy
            accuracy = ((predicted == labels).sum() / len(labels)).item()

            # compute F1 scores
            f1_scores = metrics.f1_score(
                labels, predicted, average=None, zero_division=0.0
            )
            # compute macro F1 score
            f1 = sum(f1_scores) / len(f1_scores)

            # compute precision scores
            precision_scores = metrics.precision_score(
                labels, predicted, average=None, zero_division=0.0
            )
            # compute macro precision score
            precision = sum(precision_scores) / len(precision_scores)

            # compute recall scores
            recall_scores = metrics.recall_score(
                labels, predicted, average=None, zero_division=0.0
            )
            # compute macro recall score
            recall = sum(recall_scores) / len(recall_scores)

            # compute how well the model performed on the
            # validation sets in the folds
            valid_accuracy = (
                (valid_predicted == valid_labels).sum() / len(valid_labels)
            ).item()

            # compute F1
            valid_f1_scores = metrics.f1_score(
                valid_labels, valid_predicted, average=None, zero_division=0.0
            )
            valid_f1 = sum(valid_f1_scores) / len(valid_f1_scores)

            # compute precision scores
            valid_precision_scores = metrics.precision_score(
                valid_labels, valid_predicted, average=None, zero_division=0.0
            )
            # compute macro precision score
            valid_precision = sum(valid_precision_scores) / len(valid_precision_scores)

            # compute recall scores
            valid_recall_scores = metrics.recall_score(
                valid_labels, valid_predicted, average=None, zero_division=0.0
            )
            # compute macro recall score
            valid_recall = sum(valid_recall_scores) / len(valid_recall_scores)

        return pd.DataFrame(
            {
                "accuracy": accuracy,
                "f1": f1,
                "f1_scores": [f1_scores],
                "precision": precision,
                "precision_scores": [precision_scores],
                "recall": recall,
                "recall_scores": [recall_scores],
                "valid_accuracy": valid_accuracy,
                "valid_f1": valid_f1,
                "valid_f1_scores": [valid_f1_scores],
                "valid_precision": valid_precision,
                "valid_precision_scores": [valid_precision_scores],
                "valid_recall": valid_recall,
                "valid_recall_scores": [valid_recall_scores],
            }
        )
    else:
        # split dataset
        split_data = DataSplits(
            x_data=dummy_data,
            y_data=dummy_data,
            groups=split_ids,
            train_size=0.8,
            valid_size=0.2,
            indices=split_indices,
            shuffle=True,
            random_state=data_split_seed,
        )

        # compute how well the model performs on this data split
        results = _fine_tune_transformer_for_data_split(
            num_epochs=num_epochs,
            pretrained_model_name=pretrained_model_name,
            df=df,
            feature_name=feature_name,
            label_column=label_column,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            output_dim=output_dim,
            learning_rate=learning_rate,
            seed=seed,
            loss=loss,
            gamma=gamma,
            device=device,
            batch_size=batch_size,
            path_indices=path_indices,
            split_indices=split_data.indices,
            save_model=False,
            verbose=verbose,
        )

        return pd.DataFrame(
            {
                "accuracy": results["accuracy"],
                "f1": results["f1"],
                "f1_scores": [results["f1_scores"]],
                "precision": results["precision"],
                "precision_scores": [results["precision_scores"]],
                "recall": results["recall"],
                "recall_scores": [results["recall_scores"]],
                "valid_accuracy": results["valid_accuracy"],
                "valid_f1": results["valid_f1"],
                "valid_f1_scores": [results["valid_f1_scores"]],
                "valid_precision": results["valid_precision"],
                "valid_precision_scores": [results["valid_precision_scores"]],
                "valid_recall": results["valid_recall"],
                "valid_recall_scores": [results["valid_recall_scores"]],
            }
        )


def fine_tune_transformer_average_seed(
    num_epochs: int,
    pretrained_model_name: str,
    df: pd.DataFrame,
    feature_name: str,
    label_column: str,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    output_dim: int,
    learning_rates: list[float],
    seeds: list[int],
    loss: str,
    gamma: float = 0.0,
    device: str | None = None,
    batch_size: int = 64,
    path_indices: list | np.array | None = None,
    data_split_seed: int = 0,
    split_ids: torch.Tensor | None = None,
    split_indices: tuple[Iterable[int] | None]
    | tuple[tuple[Iterable[int] | None]]
    | None = None,
    k_fold: bool = False,
    n_splits: int = 5,
    validation_metric: str = "f1",
    return_metric_for_each_fold: bool = False,
    results_output: str | None = None,
    verbose: bool = False,
):
    """
    Function to fine-tune and evaluate a model (using k-fold or standard dataset split)
    for various seeds and average over performance
    """
    if validation_metric not in ["accuracy", "f1"]:
        raise ValueError("validation_metric must be either 'accuracy' or 'f1'")

    # initialise SaveBestModel class
    model_output = f"best_bert_model_{_get_timestamp()}.pkl"
    save_best_model = SaveBestModel(
        metric=validation_metric, output=model_output, verbose=verbose
    )

    # find model parameters that has the best validation
    results_df = pd.DataFrame()
    model_id = 0

    for lr in tqdm(learning_rates):
        scores = []
        for seed in seeds:
            results = fine_tune_transformer_for_classification(
                num_epochs=num_epochs,
                pretrained_model_name=pretrained_model_name,
                df=df,
                feature_name=feature_name,
                label_column=label_column,
                label_to_id=label_to_id,
                id_to_label=id_to_label,
                output_dim=output_dim,
                learning_rate=lr,
                seed=seed,
                loss=loss,
                gamma=gamma,
                device=device,
                batch_size=batch_size,
                path_indices=path_indices,
                data_split_seed=data_split_seed,
                split_ids=split_ids,
                split_indices=split_indices,
                k_fold=k_fold,
                n_splits=n_splits,
                return_metric_for_each_fold=return_metric_for_each_fold,
                verbose=verbose,
            )

            # save metric that we want to validate on
            # take mean of performance on the folds
            # if k_fold=False, return performance for seed
            scores.append(results[f"valid_{validation_metric}"].mean())

            results["learning_rate"] = lr
            results["seed"] = seed
            results["loss_function"] = loss
            results["gamma"] = gamma
            results["k_fold"] = k_fold
            results["n_splits"] = n_splits if k_fold else None
            results["batch_size"] = batch_size
            results["model_id"] = model_id
            results_df = pd.concat([results_df, results])

        model_id += 1
        scores_mean = sum(scores) / len(scores)

        if verbose:
            print(
                f"- average{' (kfold)' if k_fold else ''} "
                f"(validation) metric score: {scores_mean}"
            )
            print(f"scores for the different seeds: {scores}")

        # save best model according to averaged metric over the different seeds
        save_best_model(
            current_valid_metric=scores_mean,
            extra_info={
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
        test_results = fine_tune_transformer_for_classification(
            num_epochs=num_epochs,
            pretrained_model_name=pretrained_model_name,
            df=df,
            feature_name=feature_name,
            label_column=label_column,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            output_dim=output_dim,
            learning_rate=checkpoint["extra_info"]["learning_rate"],
            seed=seed,
            loss=loss,
            gamma=gamma,
            device=device,
            batch_size=batch_size,
            path_indices=path_indices,
            data_split_seed=data_split_seed,
            split_ids=split_ids,
            split_indices=split_indices,
            k_fold=k_fold,
            n_splits=n_splits,
            return_metric_for_each_fold=return_metric_for_each_fold,
            verbose=verbose,
        )

        test_results["learning_rate"] = lr
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
        print(f"- average (test) metric score: {test_scores_mean}")
        print(f"scores for the different seeds: {test_scores}")

    if results_output is not None:
        print(f"saving the results dataframe to CSV in {results_output}")
        test_results_df.to_csv(results_output)

    return test_results_df
