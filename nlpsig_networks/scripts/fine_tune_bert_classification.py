from __future__ import annotations
from nlpsig import TextEncoder
import evaluate
import numpy as np
import pandas as pd
from typing import Callable, Iterable
import torch
from tqdm.auto import tqdm
from sklearn import metrics
from nlpsig.classification_utils import DataSplits, Folds
from nlpsig_networks.pytorch_utils import set_seed
from nlpsig_networks.focal_loss import FocalLoss
import os
import shutil
from datasets.arrow_dataset import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
    AdamW,
)
from typing import Iterable


def testing_transformer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
    feature_name: str,
    device: str | None = None,
    verbose: bool = False,
) -> dict[str, float | list[float]]:
    """
    Function to evaluate a transformer model by computing the accuracy
    and F1 score.
    """
    
    # set model to device is passed
    if isinstance(device, str):
        model.to(device)
    
    # loop through test set and make prediction from model
    predicted = [None for _ in range(len(test_dataset))]
    for i in tqdm(range(len(test_dataset))):
        inputs = tokenizer(test_dataset[feature_name][i],
                           return_tensors="pt")
        
        # set model to device is passed
        if isinstance(device, str):
            inputs.to(device)
            
        # obtain logits for input
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # store prediction
        predicted[i] = logits.argmax().item()

    # convert to torch tensor
    predicted = torch.tensor(predicted)
    labels = torch.tensor(test_dataset["label"])
    
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
        # print evaluation metrics
        print(
            f"Accuracy on dataset of size {len(labels)}: "
            f"{100 * accuracy} %."
        )
        print(f"- f1: {f1_scores}")
        print(f"- f1 (macro): {f1}")
        print(f"- precision (macro): {precision}")
        print(f"- recall (macro): {recall}")
        
    return {"predicted": predicted,
            "labels": labels,
            "accuracy": accuracy,
            "f1": f1,
            "f1_scores": [f1_scores],
            "precision": precision,
            "precision_scores": [precision_scores],
            "recall": recall,
            "recall_scores": [recall_scores]}


def _fine_tune_transformer_for_data_split(
    num_epochs: int,
    pretrained_model_name: str,
    df: pd.DataFrame,
    feature_name: str,
    label_column: str,
    seed: int,
    loss: str,
    gamma: float = 0.0,
    batch_size: int = 64,
    path_indices : list | np.array | None = None,
    split_indices: tuple[Iterable[int], Iterable[int], Iterable[int]] | None = None,
    save_model: bool = False,
    output_dir: str | None = None,
    device: str | None = None,
    verbose: bool = False,
) -> dict[str, float | list[float]]:
    """
    Function to fine-tune and evalaute a model for a given data_split (via split_indices)
    """
    
    # set seed
    set_seed(seed)
    
    if output_dir is None:
        output_dir = f"fine_tuned_{pretrained_model_name}"
    if not isinstance(output_dir, str):
        raise TypeError("output_dir must be either a string or None")
    
    if path_indices is not None:
        df = df.iloc[path_indices].reset_index(drop=True)

    # obtain y_data and create dictionary for converting label_to_id and id_to_label
    y_data = df[label_column]
    label_to_id = {y_data.unique()[i]: i for i in range(len(y_data.unique()))}
    id_to_label = {v: k for k, v in label_to_id.items()}
    output_dim = len(label_to_id.values())
    
    # define loss
    if loss == "focal":
        criterion = FocalLoss(gamma = gamma)
        y_train = torch.tensor(y_data[split_indices[0]].values)
        criterion.set_alpha_from_y(y=y_train)
    elif loss == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("criterion must be either 'focal' or 'cross_entropy'")
    
    # create column named "label" which are the corresponding IDs
    df["label"] = df[label_column].apply(lambda x: label_to_id[x])
    
    # initialise model, tokenizer and data_collator
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=output_dim,
        id2label=id_to_label,
        label2id=label_to_id
    )
    
    # set model to device is passed
    if isinstance(device, str):
        model.to(device)
    
    # set tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # initialise TextEncoder object from nlpsig
    # use this to fine-tune the transformer to classification task
    text_encoder = TextEncoder(df=df,
                               feature_name=feature_name,
                               model=model,
                               tokenizer=tokenizer,
                               data_collator=data_collator)
    
    # tokenize the text in df[feature_name]
    text_encoder.tokenize_text()
    
    # split the dataset using the indices which are passed in
    text_encoder.split_dataset(indices=split_indices)
    
    # set up training arguments
    text_encoder.set_up_training_args(output_dir=output_dir,
                                      num_train_epochs=num_epochs,
                                      per_device_train_batch_size=batch_size,
                                      disable_tqdm=False,
                                      save_strategy="epoch",
                                      load_best_model_at_end=True,
                                      seed=seed)
    
    # set up trainer
    def _compute_metrics(eval_pred):
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        predictions = np.argmax(eval_pred.predictions, axis=1)
        accuracy = accuracy.compute(predictions=predictions,
                                    references=eval_pred.label_ids)['accuracy']
        f1 = f1.compute(predictions=predictions, 
                        references=eval_pred.label_ids,
                        average="macro")['f1']
        return {"accuracy": accuracy, "f1": f1}

    text_encoder.set_up_trainer(data_collator=data_collator,
                                compute_metrics=_compute_metrics,
                                optimizer=AdamW(params=model.parameters(),
                                                weight_decay=0.0001),
                                custom_loss=criterion.forward)
    
    # train model
    text_encoder.fit_transformer_with_trainer_api()
    
    # evaluate 
    test_performance = testing_transformer(
        model=text_encoder.model,
        tokenizer=text_encoder.tokenizer,
        test_dataset=text_encoder.dataset_split["test"],
        feature_name=feature_name,
        device=device,
        verbose=verbose,
    )
    
    if save_model:
        text_encoder.trainer.save_model(output_dir)
    else:
        # if do not request to save the model,
        # make sure to delete any folders created
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
            
    return test_performance


def fine_tune_transformer_for_classification(
    num_epochs: int,
    pretrained_model_name: str,
    df: pd.DataFrame,
    feature_name: str,
    label_column: str,
    seed: int,
    loss: str,
    gamma: float = 0.0,
    path_indices : list | np.array | None = None,
    data_split_seed: int = 0,
    split_ids: torch.Tensor | None = None,
    split_indices: tuple[Iterable[int], Iterable[int], Iterable[int]] | tuple[tuple[Iterable[int], Iterable[int], Iterable[int]]] | None = None,
    k_fold: bool = False,
    n_splits: int = 5,
    return_metric_for_each_fold: bool = False,
    device: str | None = None,
    verbose: bool = False,
):
    """
    Function to fine-tune and evaluate a model by either using k_fold
    evaluation or a standard train/valid/test split.
    """
    
    # set seed
    set_seed(seed)
    
    # create dummy dataset for passing into Folds and DataSplit
    datasize = len(df.index) if path_indices is None else len(df.iloc[path_indices].index)
    dummy_data = torch.ones(datasize)
    
    if k_fold:
        # perform KFold evaluation and return the performance on validation and test sets
        # split dataset
        folds = Folds(x_data=dummy_data,
                      y_data=dummy_data,
                      groups=split_ids,
                      n_splits=n_splits,
                      indices=split_indices,
                      shuffle=True,
                      random_state=data_split_seed)
        
        # create lists to record the test metrics for each fold
        accuracy = []
        f1 = []
        f1_scores = []
        precision = []
        precision_scores = []
        recall = []
        recall_scores = []
        
        labels = torch.empty((0))
        predicted = torch.empty((0))
        for k in range(n_splits):
            # compute how well the model performs on this fold
            results_for_fold = _fine_tune_transformer_for_data_split(
                num_epochs=num_epochs,
                pretrained_model_name=pretrained_model_name,
                df=df,
                feature_name=feature_name,
                label_column=label_column,
                path_indices=path_indices,
                split_indices=folds.fold_indices[k],
                seed=seed,
                loss=loss,
                gamma=gamma,
                save_model=False,
                device=device,
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
            
        if not return_metric_for_each_fold:
            # compute how well the model performed on the test sets together
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
            
        return pd.DataFrame({"accuracy": accuracy,
                             "f1": f1,
                             "f1_scores": [f1_scores],
                             "precision": precision,
                             "precision_scores": [precision_scores],
                             "recall": recall,
                             "recall_scores": [recall_scores]})
    else:
        # split dataset
        split_data = DataSplits(x_data=dummy_data,
                                y_data=dummy_data,
                                groups=split_ids,
                                train_size=0.8,
                                valid_size=0.2,
                                indices=split_indices,
                                shuffle=True,
                                random_state=data_split_seed)
        
        # compute how well the model performs on this data split
        results = _fine_tune_transformer_for_data_split(
            pretrained_model_name=pretrained_model_name,
            df=df,
            feature_name=feature_name,
            label_column=label_column,
            path_indices=path_indices,
            split_indices=split_data.indices,
            seed=seed,
            loss=loss,
            gamma=gamma,
            num_epochs=num_epochs,
            save_model=False,
            device=device,
            verbose=verbose,
        )
        
        return pd.DataFrame({"accuracy": results["accuracy"],
                             "f1": results["f1"],
                             "f1_scores": [results["f1_scores"]],
                             "precision": results["precision"],
                             "precision_scores": [results["precision_scores"]],
                             "recall": results["recall"],
                             "recall_scores": [results["recall_scores"]]})
        

def fine_tune_transformer_average_seed(
    num_epochs: int,
    pretrained_model_name: str,
    df: pd.DataFrame,
    feature_name: str,
    label_column: str,
    seeds: list[int],
    loss: str,
    gamma: float = 0.0,
    path_indices : list | np.array | None = None,
    data_split_seed: int = 0,
    split_ids: torch.Tensor | None = None,
    split_indices: tuple[Iterable[int], Iterable[int], Iterable[int]] | tuple[tuple[Iterable[int], Iterable[int], Iterable[int]]] | None = None,
    k_fold: bool = False,
    n_splits: int = 5,
    validation_metric: str = "f1",
    return_metric_for_each_fold: bool = False,
    results_output: str | None = None,
    device: str | None = None,
    verbose: bool = False,
):
    """
    Function to fine-tune and evaluate a model (using k-fold or standard dataset split)
    for various seeds and average over performance
    """
    if validation_metric not in ["accuracy", "f1"]:
        raise ValueError("validation_metric must be either 'accuracy' or 'f1'")
    
    test_scores = []
    test_results_df = pd.DataFrame()
    for seed in seeds:
        test_results = fine_tune_transformer_for_classification(
            num_epochs=num_epochs,
            pretrained_model_name=pretrained_model_name,
            df=df,
            feature_name=feature_name,
            label_column=label_column,
            seed=seed,
            loss=loss,
            gamma=gamma,
            path_indices=path_indices,
            data_split_seed=data_split_seed,
            k_fold=k_fold,
            n_splits=n_splits,
            split_ids=split_ids,
            split_indices=split_indices,
            return_metric_for_each_fold=return_metric_for_each_fold,
            device=device,
            verbose=verbose,
        )
        
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
        print(f"- average (test) metric score: {test_scores_mean}")
        print(f"scores for the different seeds: {test_scores}")
        
    if results_output is not None:
        print(f"saving the results dataframe to CSV in {results_output}")
        test_results_df.to_csv(results_output)

    return test_results_df
