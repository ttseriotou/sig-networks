from __future__ import annotations

import os
import pickle

import numpy as np
import torch

from sig_networks.scripts.seqsignet_attention_encoder_functions import (
    seqsignet_attention_encoder_hyperparameter_search,
)

from ..load_anno_mi import (
    anno_mi,
    client_index,
    client_transcript_id,
    output_dim_client,
    y_data_client,
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
with open("../anno_mi_sbert.pkl", "rb") as f:
    sbert_embeddings = pickle.load(f)

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
learning_rates = [5e-4, 3e-4, 1e-4, 1e-5]
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
    "transformer_encoder_layers": 2,
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
    "path_indices": client_index,
    "split_ids": client_transcript_id,
    "k_fold": True,
    "patience": patience,
    "validation_metric": validation_metric,
    "verbose": False,
}

# run hyperparameter search
lengths = [(3, 5, 3), (3, 5, 6), (3, 5, 11), (3, 5, 26), (3, 5, 36)]

for shift, window_size, n in lengths:
    print(f"shift: {shift}, window_size: {window_size}, n: {n}")
    (
        seqsignet_attention_encoder_umap_kfold,
        best_seqsignet_attention_encoder_umap_kfold,
        _,
        __,
    ) = seqsignet_attention_encoder_hyperparameter_search(
        shift=shift,
        window_size=window_size,
        n=n,
        dim_reduce_methods=["umap"],
        results_output=f"{output_dir}/seqsignet_attention_encoder_umap_focal_{gamma}_{shift}_{window_size}_{n}_kfold.csv",
        **kwargs,
    )

    print(f"F1: {best_seqsignet_attention_encoder_umap_kfold['f1'].mean()}")
    print(
        f"Precision: {best_seqsignet_attention_encoder_umap_kfold['precision'].mean()}"
    )
    print(f"Recall: {best_seqsignet_attention_encoder_umap_kfold['recall'].mean()}")
    print(
        f"F1 scores: {np.stack(best_seqsignet_attention_encoder_umap_kfold['f1_scores']).mean(axis=0)}"
    )
    print(
        f"Precision scores: {np.stack(best_seqsignet_attention_encoder_umap_kfold['precision_scores']).mean(axis=0)}"
    )
    print(
        f"Recall scores: {np.stack(best_seqsignet_attention_encoder_umap_kfold['recall_scores']).mean(axis=0)}"
    )
