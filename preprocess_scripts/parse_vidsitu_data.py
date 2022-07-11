import os 
import pandas as pd 
import numpy as np 
import ast

csv_file="/bigdata/digbose92/VidSitu_data/csv_files/Human_verified_samples_CLIP_threshold_0_2.csv"
csv_data=pd.read_csv(csv_file)
shot_folder="/bigdata/digbose92/VidSitu_data/shot_folder"
clip_labels=list(csv_data['CLIP_labels'])
verified_labels=list(csv_data['Verified_labels'])
url=list(csv_data['Url'])
num_NA_labels=0
url_name=[]
filename_non_verified_list=[]

for i,lab in enumerate(verified_labels):
    lab=ast.literal_eval(lab)
    if((len(lab)==0) or (lab[0]=='NA')):
        num_NA_labels=num_NA_labels+1
        url_name.append(url[i])
        filename_list=url[i].split("/")
        ov_filename=os.path.join(shot_folder,filename_list[-2],filename_list[-1])
        if(os.path.exists(ov_filename)):
            filename_non_verified_list.append(ov_filename)
        #print(ov_filename)
print(len(filename_non_verified_list))

Filename_df=pd.DataFrame({'File':filename_non_verified_list})
Filename_df.to_csv("../data/Non_verified_VidSitu_samples.csv")
# print(clip_labels.shape)
# print(verified_labels.shape)
#print(csv_data.columns)