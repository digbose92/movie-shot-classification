import torch
import torch.nn as nn 
import pandas as pd 
import os 
import sys 
import time 
import pickle
#append path of datasets and models 
sys.path.append(os.path.join('..', 'datasets'))
sys.path.append(os.path.join('..', 'models'))
sys.path.append(os.path.join('..', 'configs'))
sys.path.append(os.path.join('..', 'losses'))
sys.path.append(os.path.join('..', 'optimizers'))
sys.path.append(os.path.join('..', 'utils'))
from feature_dataset import *
from baseline_models import *
import numpy as np 
import random 
from eval_model import *
import yaml
seed_value=123457
np.random.seed(seed_value) # cpu vars
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_config(config_file):

    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)
    return(config_data)

model_file="/bigdata/digbose92/MovieNet/model_dir/LSTM_video_model_movie_shot_type_scale_classification/20220703-145726_LSTM_video_model_movie_shot_type_scale_classification/20220703-145726_LSTM_best_model.pt"
model=torch.load(model_file)

#test on the entire test split and report the performance (accuracy, F1, mAP)
feature_location="/bigdata/digbose92/MovieNet/features/vit_features/feat_fps_4"
csv_file="/data/digbose92/codes/visual-scene-recognition/movie-shot-classification/data/MovieShot_split_label_file.csv"
csv_data=pd.read_csv(csv_file)
test_csv_data=csv_data[csv_data['Split']=='test']
print(len(test_csv_data))
max_len=8
batch_size=32
test_shuffle=False

###### test dataset and test dataloader declaration ###############
test_ds=LSTMDataset(folder=feature_location,
                        feat_data=test_csv_data,
                        max_len=max_len)
print(len(test_ds))

test_dl=DataLoader(test_ds,
            batch_size=batch_size,
            shuffle=test_shuffle)

######## model declaration ############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
model.eval()

####### loss criterion ###############
criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

test_accuracy,f1_score_test,precision_score_test,recall_score_test,mean_test_loss=gen_validate_score_LSTM_model(test_dl,model,device,criterion)

print('Test accuracy: %f' %(test_accuracy))
print('Test F1: %f' %(f1_score_test))
print('Test precision: %f' %(precision_score_test))
print('Test recall: %f' %(recall_score_test))
print('Test loss: %f' %(mean_test_loss))








