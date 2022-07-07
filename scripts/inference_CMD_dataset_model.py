import torch
import torch.nn as nn 
import pandas as pd 
import os 
import sys 
import time 
import timm
import pickle
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
#append path of datasets and models 
sys.path.append(os.path.join('..', 'datasets'))
sys.path.append(os.path.join('..', 'preprocess_scripts'))
sys.path.append(os.path.join('..', 'models'))
sys.path.append(os.path.join('..', 'configs'))
sys.path.append(os.path.join('..', 'losses'))
sys.path.append(os.path.join('..', 'optimizers'))
sys.path.append(os.path.join('..', 'utils'))
#from feature_model_dataset import *
from baseline_models import *
import numpy as np 
import random 
#from evaluate_video_model import *
import yaml
from tqdm import tqdm 
#from vit_feature_extraction import run_frame_wise_feature_inference

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


def run_inference_single_file(feat_data,model,device,max_len):

    padded=np.zeros((max_len,feat_data.shape[1]))
    if(feat_data.shape[0]>max_len):
        padded=feat_data[:max_len,:]
        len_sample=max_len
    else:
        padded[:feat_data.shape[0],:]=feat_data
        len_sample=feat_data.shape[0]

    feat=torch.Tensor(padded)
    feat=feat.unsqueeze(0)
    feat=feat.float()

    len_list=np.array([len_sample])
    model.eval()
    feat=feat.to(device)
    
    logits = model(feat,len_list)
    log_softmax=nn.LogSoftmax(dim=-1)
    logits=log_softmax(logits)
    y_pred = torch.max(logits, 1)[1].cpu().numpy()[0]

    return(y_pred)

#model file and pretrained data
model_file="/bigdata/digbose92/MovieNet/model_dir/LSTM_video_model_movie_shot_type_scale_classification/20220703-145726_LSTM_video_model_movie_shot_type_scale_classification/20220703-145726_LSTM_best_model.pt"
model=torch.load(model_file)
max_len=8
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)

#csv data for CLIP tags
csv_file="/data/digbose92/codes/visual-scene-recognition/movie-shot-classification/data/CLIP_file_list_tag.csv"
feature_location="/bigdata/digbose92/MovieNet/features/vit_features/feat_CMD_fps_4"
csv_data=pd.read_csv(csv_file)

#movie shot type data
pkl_file="/bigdata/digbose92/MovieNet/pkl_files/movie_shot_type.pkl"
with open(pkl_file, "rb") as f:
    movie_shot_mapping_data=pickle.load(f)

print(movie_shot_mapping_data)
#print(csv_data.shape)
num_exist_file=0
label_list=['NA']*len(csv_data)
print(len(label_list))
for i in tqdm(np.arange(csv_data.shape[0])):
    filename=csv_data['File'].iloc[i]
    npy_filename=os.path.join(feature_location,filename.split("/")[-1].split(".")[0]+".npy")
    feat_data=np.load(npy_filename)
    pred_label=run_inference_single_file(feat_data,model,device,max_len)
    #print(movie_shot_mapping_data[str(pred_label)])
    label_list[i]=movie_shot_mapping_data[str(pred_label)]

csv_data['Predicted_shot_label']=label_list
csv_data.to_csv('../data/Movieshot_CMD_predicted_stats.csv')
    # if(os.path.exists(npy_filename)):
    #     num_exist_file+=1
    
    
#print(npy_filename)
#print(num_exist_file)

#/bigdata/digbose92/Condensed_Movies/shot_folder/video_clips_shots_complete/zZlbperC3ns/zZlbperC3ns-Scene-019.mp4
# npy_file="/bigdata/digbose92/Condensed_Movies/features/vit_base/zZlbperC3ns-Scene-019.npy"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



