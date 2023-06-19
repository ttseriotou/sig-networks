import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pickle
import sys
import re
import random
import spacy
import math 

sys.path.insert(0, "../../timeline_generation/")  # Adds higher directory to python modules path
import data_handler


TalkLifeDataset = data_handler.TalkLifeDataset()
annotations = TalkLifeDataset.return_annotated_timelines(load_from_pickle=False)
annotations = annotations[annotations['content']!='nan']

sample_size = annotations.shape[0]
print(sample_size)


#post embedding
from embeddings import Representations

rep = Representations(type = 'SBERT')
embeddings_sentence = rep.get_embeddings()

print(embeddings_sentence.shape)

#dimensionality reduction
embeddings_reduced = []


#concatenate new dataframe
from dataset import get_modeling_dataframe
df = get_modeling_dataframe(annotations, embeddings_sentence, embeddings_reduced)

df = df.sort_values(by=['timeline_id', 'datetime']).reset_index(drop=True)


def padded_timeline(X, tl_ids, time_n=124, k=124, k_last=True):
    #iterate to create slices
    start_i = 0
    end_i = 0
    dims = X.shape[1]
    zeros =  np.repeat(123,dims)
    sample_list = []
    zero_padding = True

    for i in range(X.shape[0]):
        if (i==0):
            i_prev = 0
        else:
            i_prev = i-1
        if (tl_ids[i]==tl_ids[i_prev]):
            end_i +=1
            if ((k_last==True) & ((end_i - start_i) > k)):
                start_i = end_i - k
        else: 
            start_i = i
            end_i = i+1

        #data point with history
        Xtrain_add = X[start_i:end_i, :][np.newaxis, :, :]
        #padding length
        if (k_last == True):
            padding_n = k - (end_i- start_i) 
        else:
            padding_n = time_n - (end_i- start_i) 
        #create padding
        if zero_padding:
            zeros_tile = np.tile(zeros,(padding_n,1))[np.newaxis, :, :]
        #append zero padding
        Xtrain_padi =  np.concatenate((Xtrain_add, zeros_tile) ,axis=1)
        #append each sample to final list
        sample_list.append(Xtrain_padi)

    return np.concatenate(sample_list)

#max number of posts considered 
k = 29

#sbert representations
emb_str = "^e\w*[0-9]"
X = np.array(df[[c for c in df.columns if re.match(emb_str, c)]])
tl_ids = df['timeline_id'].tolist()

#padding
x_data = padded_timeline(X=X, tl_ids=tl_ids, time_n=124, k=k, k_last=True)

print(X.shape, x_data.shape)


import torch.nn as nn

class BILSTM(nn.Module):
    def __init__(self, hidden_dim_lstm1 ,hidden_dim_lstm2, output_dim, dropout_rate):
        super(BILSTM, self).__init__()

        self.hidden_dim_lstm1 = hidden_dim_lstm1
        self.hidden_dim_lstm2 = hidden_dim_lstm2
        #BiLSTMs
        self.lstm1 = nn.LSTM(input_size=384, hidden_size=hidden_dim_lstm1, num_layers=1, batch_first=True, bidirectional=True).double()
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=hidden_dim_lstm1, hidden_size=hidden_dim_lstm2, num_layers=1, batch_first=True, bidirectional=True).double()
        self.fc3 = nn.Linear(hidden_dim_lstm2, output_dim)
    
    def forward(self, x):
        seq_lengths = torch.sum(x[:, :, 0] != 123, 1)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        x = x[perm_idx]

        #BiLSTM 1
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
        out, (_, _) = self.lstm1(x_pack)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        inverse_perm = np.argsort(perm_idx)
        out = out[inverse_perm]
        out = out[:, :, :self.hidden_dim_lstm1] + out[:, :, self.hidden_dim_lstm1:]

        out = self.dropout(out)

        #BiLSTM 2
        out = out[perm_idx]
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(out, seq_lengths, batch_first=True)
        outl, (out_h, _) = self.lstm2(x_pack)
        outl, _ = torch.nn.utils.rnn.pad_packed_sequence(outl, batch_first=True)
        outl = outl[inverse_perm]
        out = out_h[-1, :, :] + out_h[-2, :, :]
        out = out[inverse_perm]

        out = self.dropout(out)

        out = self.fc3(out.float())
        return out


def testing(model, test_loader):
      model.eval()

      labels_all = torch.empty((0))
      predicted_all = torch.empty((0))

      #Test data
      with torch.no_grad():     
            # Iterate through test dataset
            for emb_t, labels_t in test_loader:

                  # Forward pass only to get logits/output
                  outputs_t = model(emb_t)

                  # Get predictions from the maximum value
                  _, predicted_t = torch.max(outputs_t.data, 1)

                  # Total correct predictions
                  labels_all = torch.cat([labels_all, labels_t])
                  predicted_all = torch.cat([predicted_all, predicted_t])
      
      return predicted_all, labels_all



from sklearn import metrics
import random
from datetime import date
import math

from classification_utils import Folds, set_seed, validation, training
from deepsignatureffn import FocalLoss

# ================================
save_results = True
# ================================

#GLOBAL MODEL PARAMETERS
model_code_name = 'SBERT_baseline_BiLSTMBiLSTMhistory_focal_3class_v3' #arch3 final focal

hidden_dim_lstm1 = [256] #[64,128,256] 
hidden_dim_lstm2 = 124 
output_dim = 3
dropout_rate = [0.25, 0.5] #[0.25, 0.5, 0.75]

num_epochs = 100
learning_rate =  [0.001, 0.0001]
BATCH_SIZE = [16, 32, 64]
gamma = [3] #[2,3]
patience = 2
k=29

# ================================
NUM_folds = 5
patience = 2
weight_decay_adam = 0.0001
RANDOM_SEED_list = [0] #[0, 1, 12, 123, 1234]


FOLDER_models = '/storage/ttseriotou/pathbert/models/v3/' 
FOLDER_results = '/storage/ttseriotou/pathbert/results/v3/' 

# ================================
KFolds = Folds(num_folds=NUM_folds)
y_data = KFolds.get_labels(df)
#y_data = KFolds.get_multilabels(df, history_len, shift=1)
# ================================
#K FOLD RUNS
ft_i = 48 #run number
for lr in learning_rate:
    for hid_d_lstm1 in hidden_dim_lstm1:
        for dp in dropout_rate:
            for b_size in BATCH_SIZE:
                for g in gamma:
                        #out_ch =  aug_l[2] 
                        str_version = 'tuning' + str(ft_i)
                        print('lr=',lr, ' hid_d_lstm1=',hid_d_lstm1,' dp=',dp, ' batch size=', b_size, 'gamma=', g)
                        ft_i+=1

                        classifier_params = {"hidden_dim_lstm1": hid_d_lstm1,
                        "hidden_dim_lstm2": hidden_dim_lstm2,
                        "output_dim": output_dim,
                        "dropout_rate": dp,
                        "num_epochs": num_epochs,
                        "learning_rate": lr,
                        "BATCH_SIZE": b_size,
                        "gamma": g,
                        "NUM_folds": NUM_folds,
                        "patience": patience,
                        #"weight_decay_adam": weight_decay_adam,
                        "total_units_k": k,
                        "RANDOM_SEED_list": RANDOM_SEED_list,
                        } 
               
                        for my_ran_seed in RANDOM_SEED_list:
                            set_seed(my_ran_seed)
                            myGenerator = torch.Generator()
                            myGenerator.manual_seed(my_ran_seed)    
                            for test_fold in range(NUM_folds):

                                print('Starting random seed #',my_ran_seed, ' and fold #', test_fold)
                                #get ith-fold data
                                x_test, y_test, x_valid, y_valid, x_train , y_train, test_tl_ids, test_pids = KFolds.get_splits(df, x_data, y_data, test_fold= test_fold)

                                #data loaders with batches

                                train = torch.utils.data.TensorDataset( torch.tensor(x_train), y_train)
                                valid = torch.utils.data.TensorDataset( torch.tensor(x_valid), y_valid)
                                test = torch.utils.data.TensorDataset( torch.tensor(x_test), y_test)

                                train_loader = torch.utils.data.DataLoader(dataset=train, batch_size = b_size, shuffle = True)
                                valid_loader = torch.utils.data.DataLoader(dataset=valid, batch_size = b_size, shuffle = True)
                                test_loader = torch.utils.data.DataLoader(dataset=test, batch_size = b_size, shuffle = True)

                    
                                #early stopping params
                                last_metric = 0
                                trigger_times = 0
                                best_metric = 0

                                #model definitions
                                model = BILSTM(hid_d_lstm1, hidden_dim_lstm2, output_dim, dp)


                                #loss function
                                alpha_values = torch.Tensor([math.sqrt(1/(y_train[y_train==0].shape[0]/y_train.shape[0])), math.sqrt(1/(y_train[y_train==1].shape[0]/y_train.shape[0])), math.sqrt(1/(y_train[y_train==2].shape[0]/y_train.shape[0]))])
                                criterion = FocalLoss(gamma = g, alpha = alpha_values)
                                #criterion = nn.CrossEntropyLoss()
                                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                                #model train/validation per epoch
                                for epoch in range(num_epochs):

                                    training(model, train_loader, criterion, optimizer, epoch, num_epochs)

                                    # Early stopping
                                    
                                    _ , f1_v, labels_val, predicted_val = validation(model, valid_loader, criterion)

                                    print('Current Macro F1:', f1_v)

                                    if f1_v > best_metric :
                                        best_metric = f1_v

                                        #test and save so far best model
                                        predicted_test, labels_test = testing(model, test_loader)

                                        results = {
                                            "model_code_name": model_code_name, 
                                            "classifier_params": classifier_params, 
                                            "date_run": date.today().strftime("%d/%m/%Y"),
                                            "test_tl_ids": test_tl_ids,
                                            "labels": labels_test,
                                            "predictions": predicted_test,
                                            "labels_val": labels_val,
                                            "predicted_val": predicted_val,
                                            "test_fold": test_fold,
                                            "random_seed": my_ran_seed,
                                            "epoch": epoch,
                                        }

                                        if (save_results==True):
                                            file_name_results = FOLDER_results + model_code_name + "_" + str(my_ran_seed) + "seed" + "_" + str(test_fold) + "fold" + "_" + str_version + '.pkl'
                                            file_name_model = FOLDER_models + model_code_name + "_" + str(my_ran_seed) + "seed" + "_" + str(test_fold) + "fold" + "_" + str_version +'.pkl'
                                            #file_name_results = FOLDER_results + model_code_name + "_" + str(my_ran_seed) + "seed" + "_" + str(test_fold) + "fold"+ '.pkl'
                                            #file_name_model = FOLDER_models + model_code_name + "_" + str(my_ran_seed) + "seed" + "_" + str(test_fold) + "fold"  +'.pkl'
                                            pickle.dump(results, open(file_name_results, 'wb'))
                                            #torch.save(model.state_dict(), file_name_model)

                                    if f1_v < last_metric:
                                        trigger_times += 1
                                        print('Trigger Times:', trigger_times)

                                        if trigger_times >= patience:
                                            print('Early stopping!')
                                            break

                                    else:
                                        print('Trigger Times: 0')
                                        trigger_times = 0

                                    last_metric = f1_v
                            
                