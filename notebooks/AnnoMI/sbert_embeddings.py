import nlpsig
import pickle
import os
from load_anno_mi import anno_mi


##################################################
########## Obtain SBERT embeddings for AnnoMI ####
##################################################

# initialise the Text Encoder
sentence_encoder = nlpsig.SentenceEncoder(df=anno_mi,
                                          feature_name="utterance_text",
                                          model_name="all-MiniLM-L12-v2")

# load in pretrained model
sentence_encoder.load_pretrained_model()

# obtain SBERT embeddings
sbert_embeddings = sentence_encoder.obtain_embeddings()

# save embeddings
path = os.path.join(os.path.dirname(__file__), "anno_mi_sbert.pkl")
with open(path, "wb") as f:
    pickle.dump(sbert_embeddings, f)
