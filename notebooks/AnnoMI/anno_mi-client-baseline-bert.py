import numpy as np
import pickle
import os
import torch
import transformers

from nlpsig_networks.scripts.fine_tune_bert_classification import (
    fine_tune_transformer_average_seed,
)

from load_anno_mi import (
    anno_mi,
    output_dim_client,
    client_index,
    client_transcript_id,
    label_to_id_client,
    id_to_label_client,
)

# set to only report critical errors to avoid excessing logging
transformers.utils.logging.set_verbosity(50)

seed = 2023

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# set output directory
output_dir = "client_talk_type_output"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# load sbert embeddings
with open("anno_mi_sbert.pkl", "rb") as f:
    sbert_embeddings = pickle.load(f)

# set features
features = ["time_encoding_minute", "timeline_index"]
standardise_method = [None, None]
include_features_in_path = True
include_features_in_input = False

# set hyperparameters
num_epochs = 5
learning_rates = [5e-5, 1e-5, 1e-6]
seeds = [1, 12, 123]
validation_metric = "f1"

# set kwargs
kwargs = {
    "num_epochs": num_epochs,
    "pretrained_model_name": "bert-base-uncased",
    "df": anno_mi,
    "feature_name": "utterance_text",
    "label_column": "client_talk_type",
    "label_to_id": label_to_id_client,
    "id_to_label": id_to_label_client,
    "output_dim": output_dim_client,
    "learning_rates": learning_rates,
    "seeds": seeds,
    "device": device,
    "batch_size": 8,
    "path_indices": client_index,
    "split_ids": client_transcript_id,
    "k_fold": True,
    "validation_metric": validation_metric,
    "verbose": False,
}

gamma = 2
for loss in ["focal", "cross_entropy"]:
    if loss == "focal":
        results_output=f"{output_dir}/bert_classifier_focal.csv",
    else:
        results_output=f"{output_dir}/bert_classifier_ce.csv"
        
    bert_classifier, best_bert_classifier, _, __ = fine_tune_transformer_average_seed(
        loss=loss,
        gamma=gamma,
        results_output=results_output,
        **kwargs,
    )
    
    print(f"F1: {best_bert_classifier['f1'].mean()}")
    print(
    f"Precision: {best_bert_classifier['precision'].mean()}"
    )
    print(f"Recall: {best_bert_classifier['recall'].mean()}")
    print(
    "F1 scores: "
    f"{np.stack(best_bert_classifier['f1_scores']).mean(axis=0)}"
    )
    print(
    "Precision scores: "
    f"{np.stack(best_bert_classifier['precision_scores']).mean(axis=0)}"
    )
    print(
    "Recall scores: "
    f"{np.stack(best_bert_classifier['recall_scores']).mean(axis=0)}"
    )
