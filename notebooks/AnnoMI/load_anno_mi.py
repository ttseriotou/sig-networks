import pandas as pd
import os


##################################################
########## Load in data ##########################
##################################################

# load in data
data_path = os.path.join(os.path.dirname(__file__), "AnnoMI-full.csv")
anno_mi = pd.read_csv(data_path)
# adding datetime column
anno_mi["datetime"] = pd.to_datetime(anno_mi["timestamp"])
# drop columns for video title and video url
anno_mi = anno_mi.drop(columns=["video_title", "video_url"])

##################################################
########## Client talk type ######################
##################################################

# obtain list of indices for when a client talks
client_index = [isinstance(x, str) for x in anno_mi["client_talk_type"]]
# obtain the y label for the client talk type
y_data_client = anno_mi["client_talk_type"][client_index]
# obtain label-to-id and id-to-label mapping
label_to_id_client = {y_data_client.unique()[i]: i
                      for i in range(len(y_data_client.unique()))}
id_to_label_client = {v: k for k, v in label_to_id_client.items()}
# convert y label to id
y_data_client = [label_to_id_client[x] for x in y_data_client]
# output dimension for client talk type
output_dim_client = len(label_to_id_client.keys())

##################################################
########## Main therapist behaviour ##############
##################################################

# obtain list of indices for when a therapist talks
therapist_index = [isinstance(x, str) for x in anno_mi["main_therapist_behaviour"]]
# obtain the y label for the therapist behaviour
y_data_therapist = anno_mi["main_therapist_behaviour"][therapist_index]
# obtain label-to-id and id-to-label mapping
label_to_id_therapist = {y_data_therapist.unique()[i]: i
                         for i in range(len(y_data_therapist.unique()))}
id_to_label_therapist = {v: k for k, v in label_to_id_therapist.items()}
# convert y label to id
y_data_therapist = [label_to_id_therapist[x] for x in y_data_therapist]
# output dimension for therapist behaviour
output_dim_therapist= len(label_to_id_therapist.keys())
