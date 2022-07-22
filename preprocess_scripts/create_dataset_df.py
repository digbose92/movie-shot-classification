import pickle
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np



# One time Processing
path = 'Insert Path of the Dataset'

df = pd.DataFrame(listdir(path+"features/vit_4_fps"),columns=["feature_file_name"])
df['path'] = df['feature_file_name'].apply(lambda x: path +"features/vit_4_fps/" + x)
with (open(path + "pkl_files/v1_v2_combined_dict.pkl", "rb")) as openfile:
    shots_data_dict = pickle.load(openfile)
    
df['trailer'] = df['feature_file_name'].apply(lambda x: x.split('_')[0])
df['shot_num'] = df['feature_file_name'].apply(lambda x: x.split('_')[2].replace(".npy",""))

def get_movement_label(x):
    return shots_data_dict[x['trailer']][x['shot_num']]['movement']['value']

def get_scale_label(x):
    if x['movement'] !=2 and x['movement'] != 3:
        return shots_data_dict[x['trailer']][x['shot_num']]['scale']['value']
    else: 
        return 5

df['movement'] = df.apply(get_movement_label,axis =1)
df['scale'] = df.apply(get_scale_label,axis =1)
df = df.set_index('feature_file_name')
df.to_csv(path +'csv_files/data.csv')