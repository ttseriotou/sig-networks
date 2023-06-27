from __future__ import annotations
from nlpsig import TextEncoder
import evaluate
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn import metrics
from nlpsig.classification_utils import DataSplits, Folds
from nlpsig_networks.pytorch_utils import set_seed
import os
from datasets.arrow_dataset import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def testing_transformer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_dataset: Dataset,
    feature_name: str
) -> dict[str, float | list[float]]:
    """
    Function to evaluate a transformer model by computing the accuracy
    and F1 score.
    """
    
    # loop through test set and make prediction from model
    predicted = [None for _ in range(len(test_dataset))]
    for i in tqdm(range(len(test_dataset))):
        inputs = tokenizer(test_dataset[feature_name][i],
                           return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted[i] = logits.argmax().item()

    # convert to torch tensor
    predicted = torch.tensor(predicted)
    labels = torch.tensor(test_dataset["label"])
    
    # compute accuracy
    accuracy = ((predicted == labels).sum() / len(labels)).item()
    # compute F1
    f1_scores = metrics.f1_score(labels, predicted, average=None)
    f1 = sum(f1_scores)/len(f1_scores)
    
    # print evaluation metrics
    print(
        f"Accuracy on dataset of size {len(labels)}: "
        f"{100 * accuracy} %."
    )
    print(f"- f1: {f1_scores}")
    print(f"- f1 (macro): {f1}")
        
    return {"predicted": predicted,
            "labels": labels,
            "accuracy": accuracy,
            "f1": f1,
            "f1_scores": f1_scores}


def _fine_tune_transformer_for_indices(
    num_epochs: int,
    pretrained_model_name: str,
    df: pd.DataFrame,
    feature_name: str,
    label_column: str,
    indices: tuple[list[int], list[int], list[int]],
    seed: int,
    save_model: bool = False,
    output_dir: str | None = None
) -> dict[str, float | list[float]]:
    """
    Function to fine-tune and evalaute a model for a given data_split (via indices)
    """
    
    # set seed
    set_seed(seed)
    
    if output_dir is None:
        output_dir = f"fine_tuned_{pretrained_model_name}"
    if not isinstance(output_dir, str):
        raise TypeError("output_dir must be either a string or None")
    
    # obtain y_data and create dictionary for converting label_to_id and id_to_label
    y_data = df[label_column]
    label_to_id = {y_data.unique()[i]: i for i in range(len(y_data.unique()))}
    id_to_label = {v: k for k, v in label_to_id.items()}
    output_dim = len(label_to_id.values())
    
    # create column named "label" which are the corresponding IDs
    df["label"] = df[label_column].apply(lambda x: label_to_id[x])
    
    # initialise model, tokenizer and data_collator
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name,
        num_labels=output_dim,
        id2label=id_to_label,
        label2id=label_to_id
    )
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
    text_encoder.split_dataset(indices=indices)
    
    # set up training arguments
    text_encoder.set_up_training_args(output_dir=output_dir,
                                      num_train_epochs=num_epochs,
                                      per_device_train_batch_size=128,
                                      disable_tqdm=False,
                                      save_strategy="steps",
                                      save_steps=10000,
                                      seed=seed)
    
    # set up trainer
    def _compute_metrics(eval_pred):
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        predictions = np.argmax(eval_pred.predictions, axis=1)
        accuracy = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)['accuracy']
        f1 = f1.compute(predictions=predictions, references=eval_pred.label_ids)['f1']
        return {"accuracy": accuracy, "f1": f1}

    text_encoder.set_up_trainer(data_collator=data_collator,
                                compute_metrics=_compute_metrics)
    
    # train model
    text_encoder.fit_transformer_with_trainer_api()
    
    # evaluate 
    test_performance = testing_transformer(
        model=text_encoder.model,
        test_dataset=text_encoder.dataset_split["test"],
        feature_name=feature_name,
    )
    
    if save_model:
        text_encoder.trainer.save_model(output_dir)
    else:
        # if do not request to save the model,
        # make sure to delete any folders created
        if os.path.isdir(output_dir):
            os.rmdir(output_dir)
            
    return test_performance


def fine_tune_transformer_for_classification(
    num_epochs: int,
    pretrained_model_name: str,
    df: pd.DataFrame,
    feature_name: str,
    label_column: str,
    seed: int,
    data_split_seed: int = 0,
    k_fold: bool = False,
    n_splits: int = 5,
    return_metric_for_each_fold: bool = False,
):
    """
    Function to fine-tune and evaluate a model by either using k_fold
    evaluation or a standard train/valid/test split.
    """
    
    # set seed
    set_seed(seed)
    
    # set y_data as the correct column
    y_data = df[label_column]
    
    if k_fold:
        # perform KFold evaluation and return the performance on validation and test sets
        # split dataset
        # x_data is just a dummy torch tensor of size (len(y_data)) to get the fold indices
        folds = Folds(x_data=torch.rand((len(y_data))),
                      y_data=torch.tensor(y_data),
                      n_splits=n_splits,
                      shuffle=True,
                      random_state=data_split_seed)
        
        accuracy = []
        f1 = []
        f1_scores = []
        labels = torch.empty((0))
        predicted = torch.empty((0))
        for k in range(n_splits):
            # compute how well the model performs on this fold
            results_for_fold = _fine_tune_transformer_for_indices(
                num_epochs=num_epochs,
                pretrained_model_name=pretrained_model_name,
                df=df,
                feature_name=feature_name,
                label_column=label_column,
                indices=folds.fold_indices[k],
                seed=seed,
                save_model=False,
            )
            
            # store the true labels and predicted labels for this fold
            labels = torch.cat([labels, results_for_fold["labels"]])
            predicted = torch.cat([predicted, results_for_fold["predicted"]])
            
            # store the metrics for this fold
            accuracy.append(results_for_fold["accuracy"])
            f1.append(results_for_fold["f1"])
            f1_scores.append(results_for_fold["f1_scores"])
            
        if return_metric_for_each_fold:
            # return how well the model performed on each individual fold
            return pd.DataFrame({"accuracy": accuracy,
                                 "f1": f1,
                                 "f1_scores": f1_scores})
        else:
            # compute how well the model performed on the test sets together
            # compute accuracy
            accuracy = ((predicted == labels).sum() / len(labels)).item()
            # compute F1
            f1_scores = metrics.f1_score(labels, predicted, average=None)
            f1 = sum(f1_scores)/len(f1_scores)
    else:
        # split dataset
        # x_data is just a dummy torch tensor of size (len(y_data)) to get the fold indices
        split_data = DataSplits(x_data=torch.rand((len(y_data))),
                                y_data=y_data,
                                train_size=0.8,
                                valid_size=0.2,
                                shuffle=True,
                                random_state=data_split_seed)
        
        # compute how well the model performs on this data split
        results = _fine_tune_transformer_for_indices(
            pretrained_model_name=pretrained_model_name,
            df=df,
            feature_name=feature_name,
            label_column=label_column,
            indices=split_data.indices,
            seed=seed,
            num_epochs=num_epochs,
            save_model=False,
        )
        
        return pd.DataFrame({"accuracy": results["accuracy"],
                             "f1": results["f1"],
                             "f1_scores": [results["f1_scores"]]})
        

def fine_tune_transformer_average_seed(
    num_epochs: int,
    pretrained_model_name: str,
    df: pd.DataFrame,
    feature_name: str,
    label_column: str,
    seeds: list[int],
    data_split_seed: int = 0,
    k_fold: bool = False,
    n_splits: int = 5,
    validation_metric: str = "f1",
    return_metric_for_each_fold: bool = False,
    results_output: str | None = None,
    verbose: bool = True
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
            data_split_seed=data_split_seed,
            k_fold=k_fold,
            n_splits=n_splits,
            return_metric_for_each_fold=return_metric_for_each_fold
        )
        
        test_results["seed"] = seed
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