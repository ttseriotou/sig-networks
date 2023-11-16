from __future__ import annotations

import os

import numpy as np
import torch
from load_rumours import df_rumours, output_dim, split_ids, y_data
from load_sbert_embeddings import sbert_embeddings

from sig_networks.scripts.lstm_baseline_functions import (
    lstm_hyperparameter_search,
)

seed = 2023

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# set output directory
output_dir = "rumours_output"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# set hyperparameters
num_epochs = 100
hidden_dim_sizes = [200, 300, 400]
num_layers = 1
bidirectional = True
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
    "df": df_rumours,
    "id_column": "timeline_id",
    "label_column": "label",
    "embeddings": sbert_embeddings,
    "y_data": y_data,
    "output_dim": output_dim,
    "hidden_dim_sizes": hidden_dim_sizes,
    "num_layers": num_layers,
    "bidirectional": bidirectional,
    "dropout_rates": dropout_rates,
    "learning_rates": learning_rates,
    "seeds": seeds,
    "loss": loss,
    "gamma": gamma,
    "device": device,
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
        bilstm_history_5_kfold,
        best_bilstm_history_5_kfold,
        _,
        __,
    ) = lstm_hyperparameter_search(
        history_lengths=[size],
        results_output=f"{output_dir}/lstm_history_{size}_focal_{gamma}_kfold.csv",
        **kwargs,
    )

    print(f"F1: {best_bilstm_history_5_kfold['f1'].mean()}")
    print(f"Precision: {best_bilstm_history_5_kfold['precision'].mean()}")
    print(f"Recall: {best_bilstm_history_5_kfold['recall'].mean()}")
    print(
        "F1 scores: "
        f"{np.stack(best_bilstm_history_5_kfold['f1_scores']).mean(axis=0)}"
    )
    print(
        "Precision scores: "
        f"{np.stack(best_bilstm_history_5_kfold['precision_scores']).mean(axis=0)}"
    )
    print(
        "Recall scores: "
        f"{np.stack(best_bilstm_history_5_kfold['recall_scores']).mean(axis=0)}"
    )
