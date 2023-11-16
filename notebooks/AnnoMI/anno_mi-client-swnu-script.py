from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from load_anno_mi import (
    anno_mi,
    client_index,
    client_transcript_id,
    output_dim_client,
    y_data_client,
)

from sig_networks.scripts.swnu_network_functions import (
    swnu_network_hyperparameter_search,
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
dimensions = [15]
swnu_hidden_dim_sizes_and_sig_depths = [([12], 3), ([10], 3)]
ffn_hidden_dim_sizes = [[32, 32], [128, 128], [512, 512]]
dropout_rates = [0.1]
learning_rates = [5e-4, 3e-4, 1e-4]
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
    "dimensions": dimensions,
    "log_signature": True,
    "pooling": "signature",
    "swnu_hidden_dim_sizes_and_sig_depths": swnu_hidden_dim_sizes_and_sig_depths,
    "ffn_hidden_dim_sizes": ffn_hidden_dim_sizes,
    "dropout_rates": dropout_rates,
    "learning_rates": learning_rates,
    "BiLSTM": True,
    "seeds": seeds,
    "loss": loss,
    "gamma": gamma,
    "device": device,
    "features": features,
    "standardise_method": standardise_method,
    "include_features_in_path": include_features_in_path,
    "include_features_in_input": include_features_in_input,
    "path_indices": client_index,
    "split_ids": client_transcript_id,
    "k_fold": True,
    "patience": patience,
    "validation_metric": validation_metric,
    "verbose": False,
}

# run hyperparameter search
lengths = [5, 11, 20, 35]

for size in lengths:
    print(f"history_length: {size}")
    (
        swnu_network_umap_kfold,
        best_swnu_network_umap_kfold,
        _,
        __,
    ) = swnu_network_hyperparameter_search(
        history_lengths=[size],
        dim_reduce_methods=["umap"],
        results_output=f"{output_dir}/swnu_network_umap_focal_{gamma}_{size}_kfold.csv",
        **kwargs,
    )

    print(f"F1: {best_swnu_network_umap_kfold['f1'].mean()}")
    print(f"Precision: {best_swnu_network_umap_kfold['precision'].mean()}")
    print(f"Recall: {best_swnu_network_umap_kfold['recall'].mean()}")
    print(
        "F1 scores: "
        f"{np.stack(best_swnu_network_umap_kfold['f1_scores']).mean(axis=0)}"
    )
    print(
        "Precision scores: "
        f"{np.stack(best_swnu_network_umap_kfold['precision_scores']).mean(axis=0)}"
    )
    print(
        "Recall scores: "
        f"{np.stack(best_swnu_network_umap_kfold['recall_scores']).mean(axis=0)}"
    )
