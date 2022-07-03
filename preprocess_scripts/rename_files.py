import os 
import shutil

src_folder="/bigdata/digbose92/MovieNet/features/vit_features/fps_4"
dest_folder="/bigdata/digbose92/MovieNet/features/vit_features/feat_fps_4"
src_file_list=os.listdir(src_folder)

for file in src_file_list:
    overall_file_path=os.path.join(src_folder,file)
    dest_file_path=file.split(".")
    destination_file_path=os.path.join(dest_folder,dest_file_path[0]+".npy")
    shutil.copy(overall_file_path,destination_file_path)