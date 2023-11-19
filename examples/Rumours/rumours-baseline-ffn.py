from __future__ import annotations

import os

import numpy as np
import torch
from load_rumours import output_dim, split_ids, y_data
from load_sbert_embeddings import sbert_embeddings

from sig_networks.scripts.ffn_baseline_functions import (
    ffn_hyperparameter_search,
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
    "x_data": sbert_embeddings,
    "y_data": y_data,
    "output_dim": output_dim,
    "hidden_dim_sizes": hidden_dim_sizes,
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

ffn_current_kfold, best_ffn_current_kfold, _, __ = ffn_hyperparameter_search(
    results_output=f"{output_dir}/ffn_current_focal_{gamma}_kfold.csv",
    **kwargs,
)

print(f"F1: {best_ffn_current_kfold['f1'].mean()}")
print(f"Precision: {best_ffn_current_kfold['precision'].mean()}")
print(f"Recall: {best_ffn_current_kfold['recall'].mean()}")
print("F1 scores: " f"{np.stack(best_ffn_current_kfold['f1_scores']).mean(axis=0)}")
print(
    "Precision scores: "
    f"{np.stack(best_ffn_current_kfold['precision_scores']).mean(axis=0)}"
)
print(
    "Recall scores: "
    f"{np.stack(best_ffn_current_kfold['recall_scores']).mean(axis=0)}"
)
