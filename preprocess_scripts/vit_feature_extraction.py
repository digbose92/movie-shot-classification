import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import os 
import cv2 
import time 
import math

activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

def run_frame_wise_feature_inference(model,transform,filename,device,dim=768,desired_frameRate=4):
    #print(filename)
    vcap=cv2.VideoCapture(filename)
    frameRate = vcap.get(5)
    intfactor=math.ceil(frameRate/desired_frameRate)
    feature_list=np.zeros((0,dim))
    frame_id=0
    length = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    tensor_list=[]
    while True:
        ret, frame = vcap.read()
        if(ret==True):
            if (frame_id % intfactor == 0):
                #print(frame_id)
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame=Image.fromarray(frame)
                tensor = transform(frame)
                tensor = tensor.to(device).unsqueeze(0) #convert each frame to tensor and pass to device 
                #tensor_list.append(tensor)
                feat_tensor=model.forward_features(tensor) #pass tensor to the model and get the features
                feat_tensor=feat_tensor.cpu().detach().numpy() #convert the feature tensors to numpy array
                feature_list=np.vstack([feature_list,feat_tensor]) #add the features to the numpy array
                del tensor
                torch.cuda.empty_cache()
            frame_id=frame_id+1
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return feature_list, frame_id

def run_frame_wise_feature_inference_stack_mode(model,transform,filename,device,dim=768,batch_size=32,desired_frameRate=4):
    #print(filename)
    vcap=cv2.VideoCapture(filename)
    frameRate = vcap.get(5)
    intfactor=math.ceil(frameRate/desired_frameRate)
    feature_list=np.zeros((0,dim))
    frame_id=0
    length = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_array_list=[]
    while True:
        ret, frame = vcap.read()
        if(ret==True):
            if (frame_id % intfactor == 0):
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame=Image.fromarray(frame)
                img_array_list.append(frame)
            frame_id=frame_id+1
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    transform_tensor=[transform(i) for i in img_array_list]
    tensor_stack=torch.stack(transform_tensor,dim=0)
    tensor_stack=tensor_stack.to(device)

    feat_array=np.zeros((0,dim))

    #need to pass batches of batch size dimension as input
    if(tensor_stack.size()[0]<=batch_size):
        feat_tensor=model.forward_features(tensor_stack)
        feat_array=feat_tensor.cpu().detach().numpy()
    else:
        num_batches=math.ceil(tensor_stack.size()[0]/batch_size)
        for i in np.arange(num_batches):
            start=i*(batch_size)
            end=(i+1)*batch_size
            if(end>tensor_stack.size()[0]):
                end=tensor_stack.size()[0]
            batch_c=tensor_stack[start:end,:,:,:]
            feat_tensor=model.forward_features(batch_c)
            feat_tensor=feat_tensor.cpu().detach().numpy()
            feat_array=np.vstack([feat_array,feat_tensor])

    del tensor_stack
    torch.cuda.empty_cache()
    return(feat_array)

print('Loading model')
model = timm.create_model('vit_base_patch16_224', pretrained=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
#model=nn.DataParallel(model) #doing data parallelism for the model to handle large batch sizes
model.eval()
# #print(model)
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

print('Loaded model')
h1 = model.pre_logits.register_forward_hook(getActivation('pre_logits'))
# #declaring the data 
feature_folder="/bigdata/digbose92/MovieNet/features/vit_features/feat_CMD_fps_4"
# feature_folder="/bigdata/digbose92/MovieNet/features/vit_features/fps_4"
csv_file="../data/CLIP_file_list_tag.csv"
#"../data/MovieShot_complete_label_file.csv"
csv_data=pd.read_csv(csv_file)['File']

# #feature nomenclature would be <id>_<shot_0042.mp4> i.e. tt5022424_shot_0042.mp4
for file in tqdm(csv_data):
    imdb_key=file.split("/")[-2]
    file_key=file.split("/")[-1]
    npy_file=os.path.join(feature_folder,file_key.split(".")[0]+".npy")
    #print(npy_file)
    if(os.path.exists(npy_file) is False):
        frame_array=run_frame_wise_feature_inference_stack_mode(model,transform,file,device)
        np.save(npy_file,frame_array)






