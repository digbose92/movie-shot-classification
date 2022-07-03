from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os 
import torch 
import torchvision.transforms as transforms
import pandas as pd 
import pickle
import numpy as np 

class LSTMDataset(Dataset):
    def __init__(self,folder,feat_data,max_len=8):

        self.folder=folder
        self.max_len=max_len 
        self.feat_data=feat_data

    def __len__(self):
        return(len(self.feat_data))

    def __getitem__(self,idx):

        filename_c=self.feat_data['Filename'].iloc[idx]
        npy_file=os.path.join(self.folder,filename_c.split("/")[-2]+"_"+filename_c.split("/")[-1].split(".")[0]+".npy")
        feat_data=np.load(npy_file)
        len_sample=feat_data.shape[0]
        feat_data=self.pad_data(feat_data)
        label=self.feat_data['Scale_value'].iloc[idx]
        

        return(feat_data,label,len_sample)

    def pad_data(self,feat_data):
        padded=np.zeros((self.max_len,feat_data.shape[1]))
        if(feat_data.shape[0]>self.max_len):
            padded=feat_data[:self.max_len,:]
        else:
            padded[:feat_data.shape[0],:]=feat_data
        return(padded)

#csv file 
csv_file="/data/digbose92/codes/visual-scene-recognition/movie-shot-classification/data/MovieShot_split_label_file.csv"
movieshot_data=pd.read_csv(csv_file)
folder="/bigdata/digbose92/MovieNet/features/vit_features/feat_fps_4"
train_movieshot_data=movieshot_data[movieshot_data['Split']=='train']
val_movieshot_data=movieshot_data[movieshot_data['Split']=='val']
test_movieshot_data=movieshot_data[movieshot_data['Split']=='test']
lstm_ds=LSTMDataset(folder=folder,feat_data=train_movieshot_data,max_len=8)
print(len(lstm_ds))
lstm_dl=DataLoader(dataset=lstm_ds,batch_size=2,shuffle=False)
vid_data,label,len_sample=next(iter(lstm_dl))
print(vid_data.size())
print(label.size())




    