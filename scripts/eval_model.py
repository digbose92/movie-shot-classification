from tqdm import tqdm 
import numpy as np 
import torch 
from statistics import mean 
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats.stats import pearsonr
import sys
import os 
sys.path.append(os.path.join('..', 'utils'))

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X, y, lengths # transpose (batch x seq_length) to (seq_length x batch)

def gen_validate_score_LSTM_model(valid_dl,model,device,criterion):
    log_softmax=nn.LogSoftmax(dim=-1)
    model.eval()
    total_correct=0
    pred_list=[]
    true=[]
    step=0
    loss_list=[]
    pred_list_mat=[]
    # pred_mat=np.array([], dtype=np.int64).reshape(0,130)
    # gt_mat=np.zeros((len(valid_dl),130))

    with torch.no_grad():
        for i, (vid_feat,label,lens) in tqdm(enumerate(valid_dl)):

            vid_feat=vid_feat.float()
            vid_feat=vid_feat.to(device)
            label=label.type(torch.LongTensor)
            label=label.to(device)

            #sort wrt lengths
            vid_feat,label,lens = sort_batch(vid_feat,label,lens)
            logits=model(vid_feat,lens.cpu().numpy())

            loss = criterion(logits,label)
            loss_list.append(loss.item())
            pred=log_softmax(logits)
            #pred_mat=np.vstack([pred_mat,pred.cpu().numpy()])
            y_pred = torch.max(pred, 1)[1]
            
            #total_correct=total_correct+(y_pred==target).sum()
            true=true+label.cpu().numpy().tolist()
            pred_list=pred_list+y_pred.cpu().numpy().tolist()

            step=step+1
            # if(step==2):
            #      break
                # print('True list:', true)
        # print('Predicted list:', pred_list)
    true=np.array(true)
    valid_accuracy=accuracy_score(true,pred_list)
    f1_score_val=f1_score(true, pred_list, average='macro')  
    precision_score_val=precision_score(true,pred_list,average='macro')
    recall_score_val=recall_score(true,pred_list,average='macro')
    mean_val_loss=mean(loss_list)

    return(valid_accuracy,f1_score_val,precision_score_val,recall_score_val,mean_val_loss)