import datetime
import json
import os

import pandas as pd
import torch

##################################################
########## Load in data ##########################
##################################################

# load in data
data_path = os.path.join(os.path.dirname(__file__), "conversations.json")
with open(data_path, "r") as f:
    data = json.load(f)

##################################################
########## Conversion to Timeline#### ############
##################################################

# Convert conversation thread to linear timeline: we use timestamps
# of each post in the twitter thread to obtain a chronologically ordered list.
def tree2timeline(conversation):
    timeline = []
    timeline.append(
        (
            conversation["source"]["id"],
            conversation["source"]["created_at"],
            conversation["source"]["stance"],
            conversation["source"]["text"],
        )
    )
    replies = conversation["replies"]
    replies_idstr = []
    replies_timestamp = []
    for reply in replies:
        replies_idstr.append(
            (reply["id"], reply["created_at"], reply["stance"], reply["text"])
        )
        replies_timestamp.append(reply["created_at"])

    sorted_replies = [x for (y, x) in sorted(zip(replies_timestamp, replies_idstr))]
    timeline.extend(sorted_replies)
    return timeline

stance_timelines = {"dev": [], "train": [], "test": []}
count_threads = 0

for subset in list(data.keys()):
    count_threads += len(data[subset])
    for conv in data[subset]:
        timeline = tree2timeline(conv)
        stance_timelines[subset].append(timeline)

##################################################
########## Obtain Dataframe ######################
##################################################
df = pd.DataFrame([], columns=["id", "label", "datetime", "text"])

total_year_hours = 365 * 24


def time_fraction(x):
    return (
        x.year
        + abs(x - datetime.datetime(x.year, 1, 1, 0)).total_seconds()
        / 3600.0
        / total_year_hours
    )


tln_idx = 0
for subset in ["train", "dev", "test"]:
    for e, thread in enumerate(stance_timelines[subset]):
        df_thread = pd.DataFrame(thread)
        df_thread = pd.DataFrame(thread, columns=["id", "datetime", "label", "text"])
        df_thread = df_thread.reindex(columns=["id", "label", "datetime", "text"])

        df_thread["timeline_id"] = str(tln_idx)
        df_thread["set"] = subset
        df_thread["id"] = df_thread["id"].astype("float64")
        df_thread["datetime"] = pd.to_datetime(df_thread["datetime"])
        df_thread["datetime"] = df_thread["datetime"].map(
            lambda t: t.replace(tzinfo=None)
        )

        df = pd.concat([df, df_thread])
        tln_idx += 1

df = df.reset_index(drop=True)

##################################################
########## Label Mapping #########################
##################################################
#labels in numbers
dictl = {}
dictl['support'] = 0
dictl['deny'] = 1
dictl['comment'] = 2
dictl['query'] = 3

#Get the numerical labels
df = df.replace({"label": dictl})

##################################################
########## Dimensions and Y labels ###############
##################################################
output_dim = len(df["label"].unique())
y_data = torch.tensor(df["label"].astype(float).values, dtype=torch.int64)