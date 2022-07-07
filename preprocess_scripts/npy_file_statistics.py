import os 
import pandas as pd 
import numpy as np 
from statistics import mean, median
from tqdm import tqdm

folder="/bigdata/digbose92/MovieNet/features/vit_features/feat_fps_4"
npy_file_list=os.listdir(folder)
shape_data=[]
for file in tqdm(npy_file_list):
    np_filename=os.path.join(folder,file)
    np_data=np.load(np_filename)
    shape_data.append(np_data.shape[0])

#mean shape: 6.67
#median shape: 5
# print(np.percentile(shape_data,75)) #8
# num_shapes=[s for s in shape_data if s>=8]
# print(len(num_shapes))
print(min(shape_data))
# print(mean(shape_data),median(shape_data))
