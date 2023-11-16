from __future__ import annotations

import os

import numpy as np
import torch
from load_rumours import df_rumours, output_dim, split_ids, y_data
from load_sbert_embeddings import sbert_embeddings

from sig_networks.scripts.swmhau_network_functions import (
    swmhau_network_hyperparameter_search,
)

seed = 2023

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# set output directory
output_dir = "rumours_output"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# set features
features = ["time_encoding", "timeline_index"]
standardise_method = ["z_score", None]
include_features_in_path = True
include_features_in_input = True

# set hyperparameters
num_epochs = 100
dimensions = [15]
# define swmhau parameters: (output_channels, sig_depth, num_heads)
swmhau_parameters = [(12, 3, 10), (10, 3, 5)]
num_layers = [1]
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
    "df": df_rumours,
    "id_column": "timeline_id",
    "label_column": "label",
    "embeddings": sbert_embeddings,
    "y_data": y_data,
    "output_dim": output_dim,
    "dimensions": dimensions,
    "log_signature": True,
    "pooling": "signature",
    "swmhau_parameters": swmhau_parameters,
    "num_layers": num_layers,
    "ffn_hidden_dim_sizes": ffn_hidden_dim_sizes,
    "dropout_rates": dropout_rates,
    "learning_rates": learning_rates,
    "seeds": seeds,
    "loss": loss,
    "gamma": gamma,
    "device": device,
    "features": features,
    "standardise_method": standardise_method,
    "include_features_in_path": include_features_in_path,
    "include_features_in_input": include_features_in_input,
    "split_ids": split_ids,
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
        swmhau_network_umap_kfold,
        best_swmhau_network_umap_kfold,
        _,
        __,
    ) = swmhau_network_hyperparameter_search(
        history_lengths=[size],
        dim_reduce_methods=["umap"],
        results_output=f"{output_dir}/swmhau_network_umap_focal_{gamma}_{size}_kfold.csv",
        **kwargs,
    )

    print(f"F1: {best_swmhau_network_umap_kfold['f1'].mean()}")
    print(f"Precision: {best_swmhau_network_umap_kfold['precision'].mean()}")
    print(f"Recall: {best_swmhau_network_umap_kfold['recall'].mean()}")
    print(
        "F1 scores: "
        f"{np.stack(best_swmhau_network_umap_kfold['f1_scores']).mean(axis=0)}"
    )
    print(
        "Precision scores: "
        f"{np.stack(best_swmhau_network_umap_kfold['precision_scores']).mean(axis=0)}"
    )
    print(
        "Recall scores: "
        f"{np.stack(best_swmhau_network_umap_kfold['recall_scores']).mean(axis=0)}"
    )
