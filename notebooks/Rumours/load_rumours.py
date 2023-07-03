import pandas as pd 
import os
import json
import datetime
import torch


##################################################
########## Load in data ##########################
##################################################

# load in data
data_path = os.path.join(os.path.dirname(__file__), "conversations.json")
with open(data_path, 'r') as f:
    data = json.load(f)


##################################################
########## Conversion to Longitudinal ############
##################################################

#Convert conversation thread to linear timeline: we use timestamps of each post in the twitter thread to obtain a chronologically ordered list.
def tree2timeline(conversation):
    timeline = []    
    timeline.append((conversation['source']['id'],conversation['source']['created_at'],conversation['source']['stance'],conversation['source']['text']))
    replies = conversation['replies']
    replies_idstr = []
    replies_timestamp = []
    for reply in replies:
        replies_idstr.append((reply['id'],reply['created_at'],reply['stance'],reply['text']))
        replies_timestamp.append(reply['created_at'])
    
    sorted_replies = [x for (y,x) in sorted(zip(replies_timestamp,replies_idstr))]
    timeline.extend(sorted_replies)
    return timeline

stance_timelines = {'dev':[],'train':[],'test':[]}
switch_timelines = {'dev':[],'train':[],'test':[]}
check = []
count_switch_threads = 0
all_support_switches = 0
all_oppose_switches = 0
count_threads = 0

for subset in  list(data.keys()):
  count_threads += len(data[subset])
  for conv in data[subset]:
    timeline = tree2timeline(conv)
    stance_timelines[subset].append(timeline)
    support = 0
    deny = 0
    old_sum = 0
    switch_events = []
    for i, s  in enumerate(timeline): 
        if s[2] == "support":
            support = support+1
        elif s[2]=="query" or s[2]=="deny":
            deny = deny+1
        
        new_sum = support - deny
        check.append(new_sum)
        
             
        if i!=0 and old_sum==0 and new_sum!=0:
            # A switch in stance from supporting to opposing the claim starts
            if new_sum<0:
              switch_events.append((s[0],s[1],-1,s[3]))
            # A switch in stance from opposing to supporting the claim starts
            elif new_sum>0:
              switch_events.append((s[0],s[1],1,s[3]))
        elif i!=0 and old_sum<0 and new_sum<0 and -1 in [x[2] for x in switch_events]:
            # A switch in stance from supporting to opposing the claim continues
            switch_events.append((s[0],s[1],-2,s[3]))
        elif i!=0 and old_sum>0 and new_sum>0 and 1 in [x[2] for x in switch_events]:
            # A switch in stance from opposing to supporting the claim continues
            switch_events.append((s[0],s[1],2,s[3]))
        
        else:
            switch_events.append((s[0],s[1],0,s[3]))
        old_sum = new_sum


    support_switch = [x[2] for x in switch_events].count(1)
    oppose_switch = [x[2] for x in switch_events].count(-1)


    if (support_switch + oppose_switch>0):
        count_switch_threads = count_switch_threads+1
        all_support_switches += support_switch
        all_oppose_switches += oppose_switch

    switch_timelines[subset].append(switch_events)

##################################################
########## Label Converstion to Binary ###########
##################################################
def simplify_label(y):
    #If the label is -2,-1,2 this is is relabeled to 1
    if y != 0:
        y = 1   
    return y    

for subset in ['train','dev','test']:
    for i,thread in enumerate(switch_timelines[subset]):
        switch_timelines[subset][i] = [(x,z,simplify_label(y),u) for (x,z,y,u) in thread]

##################################################
########## Obtain Dataframe ######################
##################################################
df_rumours = pd.DataFrame([], columns=['id','label','datetime','text'])

total_year_hours = 365*24

def time_fraction(x):
        return x.year + abs(x - datetime.datetime(x.year, 1,1,0)).total_seconds() / 3600.0 / total_year_hours

tln_idx = 0
for subset in ['train','dev','test']:
    for e, thread in enumerate(switch_timelines[subset]):
        df_thread = pd.DataFrame(thread)
        df_thread = pd.DataFrame(thread, columns = ['id', 'datetime','label','text'])
        df_thread = df_thread.reindex(columns=['id','label','datetime','text'])

        df_thread['timeline_id'] = str(tln_idx)
        df_thread['id'] = df_thread['id'].astype('float64')
        df_thread['datetime']= pd.to_datetime(df_thread['datetime'])
        df_thread['datetime'] = df_thread['datetime'].map(lambda t: t.replace(tzinfo=None))
        #df_thread['timestamp'] = df_thread['datetime'].map(lambda t: time_fraction(t))

        df_rumours = pd.concat([df_rumours, df_thread])
        tln_idx+=1

df_rumours = df_rumours.reset_index(drop=True)

##################################################
########## Dimensions and Y labels ###############
##################################################
output_dim = len(df_rumours['label'].unique())
y_data = torch.tensor(df_rumours['label'].astype(float).values, dtype=torch.int64)