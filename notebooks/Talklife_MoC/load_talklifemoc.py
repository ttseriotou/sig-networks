import pandas as pd 
import sys
import pickle
import torch
import paths

sys.path.append("..")# Adds higher directory to python modules path
import utils.data_handler as data_handler

##################################################
########## Load in data ##########################
##################################################
#TalkLifeDataset = data_handler.TalkLifeDataset()
#df = TalkLifeDataset.return_annotated_timelines(load_from_pickle=True)
df = pd.read_pickle(paths.data_fname)
df = df[df['content']!='nan']

#labels in numbers
dictl = {}
dictl['0'] = '0'
dictl['IE'] = '1'
dictl['IEP'] = '1'
dictl['ISB'] = '2'
dictl['IS'] = '2'#

#GET THE FLAT LABELS
df = df.replace({"label": dictl})

#read pickle of folds
folds_fname = paths.folds_fname
with open(folds_fname, 'rb') as f:
    folds = pickle.load(f)

#obtain columns for train/dev/test for each fold
for fold in folds.keys():
    df['fold'+str(fold)] = df['timeline_id'].map(lambda x: 'train' if x in folds[fold]['train'] else ('dev' if x in folds[fold]['dev'] else 'test'))

#rest index
df = df.reset_index(drop=True)
##################################################
########## Dimensions and Y labels ###############
##################################################
output_dim = len(df['label'].unique())
y_data = torch.tensor(df['label'].astype(float).values, dtype=torch.int64)