import pickle
import numpy as np
import pandas as pd
import re
import torch
import os

#read data
cwd = os.getcwd()
sbert_file = cwd+'/notebooks/Rumours/sbert.pkl'
with open(sbert_file,'rb') as g:
    emb_data = pickle.load(g)

def embedding_df(emb_data):
    #embeddings in df
    row_list = []

    for subset in ['train', 'dev', 'test']:
        for thread in emb_data[subset]:
            record = np.concatenate((np.array(thread['source']['id']).reshape(1,), np.array(subset).reshape(1,),thread['source']['emb']))
            row_list.append(record)

            for tweet in thread['replies']:
                record = np.concatenate((np.array(tweet['id']).reshape(1,), np.array(subset).reshape(1,), tweet['emb']))
                row_list.append(record)
        
    df_emb = pd.DataFrame(row_list)               
    df_emb.columns =['id', 'subset']+ ['e' + str(i+1) for i in range(emb_data['train'][0]['replies'][0]['emb'].shape[0])]
    df_emb['id'] = df_emb['id'].astype('float64')
    df_emb[[c for c in df_emb.columns if re.match("^e\w*[0-9]", c)]]= df_emb[[c for c in df_emb.columns if re.match("^e\w*[0-9]", c)]].astype('float')

    df_emb = df_emb.reset_index(drop=True)

    return df_emb

df_emb = embedding_df(emb_data)

#ALIGN THE EMBEDDINGS WITH THE RUMOUR TWEETS BY ID
with open("notebooks/Rumours/load_rumours.py") as f:
    exec(f.read()) 

df_emb = df_rumours[['id']].merge(df_emb, on="id", how="left")
df_emb = df_emb.reset_index(drop=True)

sbert_embeddings = torch.tensor(df_emb[[c for c in df_emb.columns if re.match("^e\w*[0-9]", c)]].values)