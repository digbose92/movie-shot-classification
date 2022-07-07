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
from vit_feature_extraction import run_frame_wise_feature_inference

seed_value=123457
np.random.seed(seed_value) # cpu vars
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook


def load_config(config_file):

    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)
    return(config_data)

model_file="/bigdata/digbose92/MovieNet/model_dir/LSTM_video_model_movie_shot_type_scale_classification/20220703-145726_LSTM_video_model_movie_shot_type_scale_classification/20220703-145726_LSTM_best_model.pt"
model=torch.load(model_file)
max_len=8
#/bigdata/digbose92/Condensed_Movies/shot_folder/video_clips_shots_complete/zZlbperC3ns/zZlbperC3ns-Scene-019.mp4
npy_file="/bigdata/digbose92/Condensed_Movies/features/vit_base/zZlbperC3ns-Scene-019.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if(os.path.exists(npy_file) is False):
    #run vit feature extraction here
    print("Loading Vit model")
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
    vit_model=vit_model.to(device)
    vit_model.eval()
    config = resolve_data_config({}, model=vit_model)
    transform = create_transform(**config)
    h1 = vit_model.pre_logits.register_forward_hook(getActivation('pre_logits'))
    key=npy_file.split("/")[-1].split(".")[0]
    scene_index=key.index("Scene")
    subfolder_name=key[0:scene_index-1]
    video_filename=os.path.join("/bigdata/digbose92/Condensed_Movies/shot_folder/video_clips_shots_complete",subfolder_name,npy_file.split("/")[-1].split(".")[0]+".mp4")
    #print(video_filename)
    feat_data, frame_list= run_frame_wise_feature_inference(vit_model,transform,video_filename,device,dim=768,desired_frameRate=4)
    print('Run feature extraction here')
else:
    feat_data=np.load(npy_file)
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
model=model.to(device)

logits = model(feat,len_list)
log_softmax=nn.LogSoftmax(dim=-1)
logits=log_softmax(logits)
y_pred = torch.max(logits, 1)[1].cpu().numpy()[0]

pkl_file="/bigdata/digbose92/MovieNet/pkl_files/movie_shot_type.pkl"
with open(pkl_file, "rb") as f:
    movie_shot_data=pickle.load(f)

#print(movie_shot_data)
print(movie_shot_data[str(y_pred)])





