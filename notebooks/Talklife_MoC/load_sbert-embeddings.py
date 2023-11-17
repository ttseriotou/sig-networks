import pickle
import torch
import os
import paths

# read embeddings
emb_sbert_filename = paths.embeddings_fname
with open(emb_sbert_filename, "rb") as f:
    sbert_embeddings = pickle.load(f)

sbert_embeddings = torch.tensor(sbert_embeddings)
