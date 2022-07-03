#initial scripts to parse movie shot data including shot duration, labels and distribution
import os 
import json 
import pandas as pd 
import numpy as np 
import json
import pickle
#def combine_data(v1_data,v2_data,v3_data):

def combine_v1_v2_data(v1_data,v2_data):
    #append the dictionary to the one associated with individual key in v1_data for common keys between v1 and v2
    v1_keys=list(v1_data.keys())
    v2_keys=list(v2_data.keys())

    v1_intersect_v2=list(set(v1_keys) & set(v2_keys))
    v1_only_keys=list(set(v1_keys) - set(v2_keys))
    v2_only_keys=list(set(v2_keys) - set(v1_keys))

    # print(len(v2_keys),len(v1_keys))
    # print(len(v1_intersect_v2),len(v1_only_keys),len(v2_only_keys))

    dict_tot={}

    for key in v1_only_keys:
        dict_tot[key]=v1_data[key]

    for key in v2_only_keys:
        dict_tot[key]=v2_data[key]

    for key in v1_intersect_v2:
        dict_c=v1_data[key]
        dict_c_2=v2_data[key]

        #append both of them
        z = {**dict_c, **dict_c_2}
        dict_tot[key]=z

    return(dict_tot)

base_folder="/bigdata/digbose92/MovieNet/"
trailer_shot_types=os.path.join(base_folder,'trailer_shot_types')
trailer_videos=os.path.join(base_folder,'trailer')

num_videos=0
file_list=[]
movie_id_list=os.listdir(trailer_videos)
# movie_id_list=list(set(trailer_videos))
print(len(movie_id_list),len(set(movie_id_list)))

for movie_id in os.listdir(trailer_videos):
    sub_fold=os.path.join(trailer_videos,movie_id)
    mp4_file_list=os.listdir(sub_fold)
    file_list=file_list+[os.path.join(sub_fold,x) for x in mp4_file_list]

# with open()
#print(len(file_list)) #34259 videos 

#now match with the label list
v1_data=json.load(open('/bigdata/digbose92/MovieNet/trailer_shot_types/v1_full_trailer.json','r'))
v2_data=json.load(open('/bigdata/digbose92/MovieNet/trailer_shot_types/v2_full_trailer.json','r'))
v3_data=json.load(open('/bigdata/digbose92/MovieNet/trailer_shot_types/v3_full.json'))['full']['trailer']

#print(v3_data)
v1_keys=list(v1_data.keys())
v2_keys=list(v2_data.keys())
v3_keys=list(v3_data.keys())

v1_intersect_v2=list(set(v1_keys) & set(v2_keys))
v1_intersect_v3=list(set(v1_keys) & set(v3_keys))

total_keys=v1_keys+v2_keys+v3_keys
total_keys=list(set(total_keys))


#check v1 and v2 have same data in the intersecting keys
num_common_v1_v2=0
# for key in v1_intersect_v2:
#     print(key)
#     print('V1_data:', (v1_data[key]))
#     print('V2_data:', (v2_data[key]))
    # if(v2_data[key].items()<=v1_data[key].items()):
    #     num_common_v1_v2+=1

# num_common_v1_v3=0
#for key in v1_intersect_v3:
    # if(v1_data[key]==v3_data[key]):
    #     num_common_v1_v3+=1
# print(len(v1_intersect_v3),num_common_v1_v3)
# print(len(v1_intersect_v2),num_common_v1_v2)

#print(len(total_keys))
# print(len(v1_intersect_v2),len(v1_intersect_v3))
# print(len(v1_keys),len(v2_keys),len(v3_keys))

v1_v2_combined_dict=combine_v1_v2_data(v1_data,v2_data)

print(v1_v2_combined_dict)
#save it 
with open('/bigdata/digbose92/MovieNet/pkl_files/v1_v2_combined_dict.pkl','wb') as f:
    pickle.dump(v1_v2_combined_dict,f)

#have to combine v1 v2 and v3 
#print(len(v1_data),len(v2_data),len(v3_data))
#csv file with columns as ['file'],['Scale_label'] ['Scale_value'],['Movement_label'],['Movement_value']

scale_label_list=[]
scale_value_list=[]
movement_label_list=[]
movement_value_list=[]

for file in file_list:
    imdbid=file.split("/")[-2]
    shot_key=file.split("/")[-1].split("_")[-1].split(".")[0]
    #print(imdbid,shot_key)
    shot_info_data=v1_v2_combined_dict[imdbid]

    if(shot_key in shot_info_data):
        #print(shot_info_data[shot_key]) #sample:{'scale': {'label': 'CS', 'value': 1}, 'movement': {'label': 'Motion', 'value': 0}}
        #print(shot_key,shot_info_data[shot_key]c)
        c_data=shot_info_data[shot_key]
        if('scale' in c_data):
            scale_label_list.append(shot_info_data[shot_key]['scale']['label'])
            scale_value_list.append(shot_info_data[shot_key]['scale']['value'])
        else:
            scale_label_list.append('NA')
            scale_value_list.append('NA')

        if('movement' in c_data):
            movement_label_list.append(shot_info_data[shot_key]['movement']['label'])
            movement_value_list.append(shot_info_data[shot_key]['movement']['value'])
        else:
            movement_label_list.append('NA')
            movement_value_list.append('NA')


df_val=pd.DataFrame({'Filename':file_list,
                    'Scale_label':scale_label_list,
                    'Scale_value':scale_value_list,
                    'Movement_label':movement_label_list,
                    'Movement_value':movement_value_list})

df_val.to_csv('../data/MovieShot_complete_label_file.csv')
#     #print(shot_key)
#     #print(imdbid,shot_id
#     #print(imdbid)

#print(v1_data['tt3624740'])
