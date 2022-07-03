import os 
import pandas as pd 
import numpy as np 
from collections import Counter
from random import sample

csv_file="/data/digbose92/codes/visual-scene-recognition/movie-shot-classification/data/MovieShot_complete_label_file.csv"
movie_shot_data=pd.read_csv(csv_file)
movie_shot_not_scale_data = movie_shot_data[movie_shot_data['Scale_value'].notna()]
scale_value_list=list(movie_shot_not_scale_data['Scale_value'])

#train/val/test (70/20/10) 
split=0.70
id_list=list(np.arange(len(movie_shot_not_scale_data)))
train_id_list_shots=sample(id_list,int(split*len(id_list)))
diff_movie_list=set(id_list).difference(set(train_id_list_shots))
val_id_list_shots=sample(list(diff_movie_list),int(0.67*len(diff_movie_list)))
test_id_list_shots=diff_movie_list.difference(set(val_id_list_shots))
test_id_list_shots=list(test_id_list_shots)

#print(len(train_id_list_shots),len(test_id_list_shots),len(val_id_list_shots))
split_list=['split']*len(movie_shot_not_scale_data)

#train split insertion here 
for i,sp in enumerate(train_id_list_shots):
    split_list[train_id_list_shots[i]]='train'

#val split insertion here
for i,sp in enumerate(val_id_list_shots):
    split_list[val_id_list_shots[i]]='val'

#test split insertion here
for i,sp in enumerate(test_id_list_shots):
    split_list[test_id_list_shots[i]]='test'

#print(Counter(split_list))
movie_shot_not_scale_data['Split']=split_list
movie_shot_not_scale_data.to_csv("../data/MovieShot_split_label_file.csv")

