import numpy as np
import pickle
import os
import torch
from nlpsig_networks.scripts.ffn_baseline_functions import (
    histories_baseline_hyperparameter_search,
)
from load_anno_mi import (
    anno_mi,
    y_data_client,
    output_dim_client,
    client_index,
    client_transcript_id,
)

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
num_epochs = 100
hidden_dim_sizes = [[64, 64], [128, 128], [256, 256], [512, 512]]
dropout_rates = [0.1]
learning_rates = [1e-3, 5e-4, 1e-4]
seeds = [1, 12, 123]
loss = "focal"
gamma = 2
validation_metric = "f1"
patience = 3

# set kwargs
kwargs = {
    "num_epochs": num_epochs,
    "df": anno_mi,
    "id_column": "transcript_id",
    "label_column": "client_talk_type",
    "embeddings": sbert_embeddings,
    "y_data": y_data_client,
    "output_dim": output_dim_client,
    "hidden_dim_sizes": hidden_dim_sizes,
    "dropout_rates": dropout_rates,
    "learning_rates": learning_rates,
    "seeds": seeds,
    "loss": loss,
    "gamma": gamma,
    "device": device,
    "path_indices": client_index,
    "split_ids": client_transcript_id,
    "k_fold": True,
    "patience": patience,
    "validation_metric": validation_metric,
    "verbose": False,
}

(
    ffn_mean_history_kfold,
    best_ffn_mean_history_kfold,
    _,
    __,
) = histories_baseline_hyperparameter_search(
    use_signatures=False,
    results_output=f"{output_dir}/ffn_mean_history_focal_{gamma}_kfold.csv",
    **kwargs,
)

print(f"F1: {best_ffn_mean_history_kfold['f1'].mean()}")
print(
    f"Precision: {best_ffn_mean_history_kfold['precision'].mean()}"
)
print(f"Recall: {best_ffn_mean_history_kfold['recall'].mean()}")
print(
    "F1 scores: "
    f"{np.stack(best_ffn_mean_history_kfold['f1_scores']).mean(axis=0)}"
)
print(
    "Precision scores: "
    f"{np.stack(best_ffn_mean_history_kfold['precision_scores']).mean(axis=0)}"
)
print(
    "Recall scores: "
    f"{np.stack(best_ffn_mean_history_kfold['recall_scores']).mean(axis=0)}"
)
