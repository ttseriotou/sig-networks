import pandas as pd
import os
import datetime
import torch
import paths

##################################################
########## Load in data ##########################
##################################################
reddit_fname = paths.data_fname
df = pd.read_pickle(reddit_fname)

dict3 = {}
dict3["0"] = "0"
dict3["E"] = "1"
dict3["IE"] = "1"
dict3["S"] = "2"
dict3["IS"] = "2"

# GET THE FLAT LABELS
df = df.replace({"label_3": dict3})
df["label"] = df["label_3"]

# GET TRAIN/DEV/TEST SET
df["set"] = None
df.loc[(df.train_or_test == "test"), "set"] = "test"
df.loc[(df.train_or_test != "test") & (df.fold != 0), "set"] = "train"
df.loc[(df.fold == 0), "set"] = "dev"

# KEEP SPECIFIC COLUMNS AND RESET INDEX
df = df[
    ["user_id", "timeline_id", "postid", "content", "label", "datetime", "set"]
].reset_index(drop=True)

##################################################
########## Dimensions and Y labels ###############
##################################################
output_dim = len(df["label"].unique())
y_data = torch.tensor(df["label"].astype(float).values, dtype=torch.int64)
