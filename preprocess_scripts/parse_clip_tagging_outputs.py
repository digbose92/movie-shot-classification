import os 
import pickle 
from tqdm import tqdm
import pandas as pd 

folder="/bigdata/digbose92/Condensed_Movies/video_clips_shots_tags_complete"
file_list=os.listdir(folder)
src_shot_folder="/bigdata/digbose92/Condensed_Movies/shot_folder/video_clips_shots_complete"
#zzkjLhitH7c.pkl"

# with open(sample_file,"rb") as f:
#     sample_data=pickle.load(f)

#print(sample_data['zzkjLhitH7c-Scene-002.mp4']) #interrogation room with extreme closeup
#/bigdata/digbose92/Condensed_Movies/shot_folder/video_clips_shots_complete/zzkjLhitH7c/zzkjLhitH7c-Scene-004.mp4 -> funeral with extreme closeup
num_files=0
file_list_save=[]
top_1_score_save=[]

for subfold in tqdm(file_list):
    fold_path=os.path.join(folder,subfold)
    shot_subfold=os.path.join(src_shot_folder,subfold.split(".")[0])
    with open(fold_path,"rb") as f:
        data=pickle.load(f)
    for key in list(data.keys()):
        if (len(data[key]['Values'])>0):
            if(data[key]['Values'][0]<=0.2):
                filepath=os.path.join(shot_subfold,key)
                if(os.path.exists(filepath)):
                    #print(filepath)
                    file_list_save.append(filepath)
                    top_1_score_save.append(data[key]['Values'][0])
                    num_files=num_files+1

clip_file_list_tag=pd.DataFrame({'File':file_list_save,'Top_1_score':top_1_score_save})
clip_file_list_tag.to_csv("../data/CLIP_file_list_tag.csv")
#print(num_files) #745059
# print(data[key]['Values'])
