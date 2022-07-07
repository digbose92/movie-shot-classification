import pandas as pd 
import numpy as np 
import os 
import sys 
import time 
import pickle
#append path of datasets and models 
sys.path.append(os.path.join('..', 'datasets'))
sys.path.append(os.path.join('..', 'models'))
sys.path.append(os.path.join('..', 'configs'))
sys.path.append(os.path.join('..', 'losses'))
sys.path.append(os.path.join('..', 'optimizers'))
sys.path.append(os.path.join('..', 'utils'))

import random
from ast import literal_eval
import torch
import yaml
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from feature_dataset import *
from loss_functions import *
from baseline_models import *
from optimizer import *
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from eval_model import gen_validate_score_LSTM_model
#from metrics import * #not using metrics for now - going for a standard metrics here 
from tqdm import tqdm 
from statistics import mean
import argparse
from log_file_generate import *
from scipy.stats.stats import pearsonr
import wandb

seed_value=123457
np.random.seed(seed_value) # cpu vars
torch.manual_seed(seed_value) # cpu  vars
random.seed(seed_value) # Python
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X, y, lengths # transpose (batch x seq_length) to (seq_length x batch)

def load_config(config_file):
    with open(config_file,'r') as f:
        config_data=yaml.safe_load(f)
    return(config_data)

def main(config_data):

    feature_location=config_data['data']['feature_folder']
    csv_data=pd.read_csv(config_data['data']['csv_file'])

    #train and val split from the data 
    train_csv_data=csv_data[csv_data['Split']=='train']
    val_csv_data=csv_data[csv_data['Split']=='val']

    #extracting the parameters (batch size, epochs, max len)
    batch_size=config_data['parameters']['batch_size']
    epochs=config_data['parameters']['epochs']
    max_len=config_data['parameters']['max_len']
    train_shuffle=config_data['parameters']['train_shuffle']
    val_shuffle=config_data['parameters']['val_shuffle']

    ################### declaring the dataset and dataloaders here #############
    train_ds=LSTMDataset(folder=feature_location,
                        feat_data=train_csv_data,
                        max_len=max_len)

    val_ds=LSTMDataset(folder=feature_location,
                        feat_data=val_csv_data,
                        max_len=max_len)

    train_dl=DataLoader(train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle)

    val_dl=DataLoader(val_ds,
            batch_size=batch_size,
            shuffle=val_shuffle)

    #define the device here
    if(config_data['device']['is_cuda']):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #################################### WANDB INITIALIZATION ########################
    wandb.login()
    wandb.init(project="movie shot type classification", entity="digbwb", config=config_data)

    model=LSTM_model(config_data['model']['embedding_dim'],
                    config_data['model']['n_hidden'],
                    config_data['model']['n_classes'],
                    config_data['model']['n_layers'],
                    config_data['model']['batch_first'])

    model=model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of parameters: %d' %(params))

    if(config_data['loss']['loss_option']=='cross_entropy_loss'):
        criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    if(config_data['optimizer']['choice']=='Adam'):
        optim_example=optimizer_adam(model,float(config_data['optimizer']['lr']))

    #using placeholder for now -- add later
    if(config_data['optimizer']['scheduler']=='step_lr_plateau'):
        lr_scheduler=reducelr_plateau(optim_example,mode=config_data['optimizer']['mode'],factor=config_data['optimizer']['factor'],patience=config_data['optimizer']['patience'],
        verbose=config_data['optimizer']['verbose'])

    if(config_data['optimizer']['scheduler']=='step_lr'):
        lr_scheduler=steplr_scheduler(optim_example,
                        step_size=config_data['optimizer']['step_size'],
                        gamma=config_data['optimizer']['gamma'])

    max_epochs=config_data['parameters']['epochs']
    print('Starting training')

    best_models=config_data['output']['model_dir']

    #create a folder with each individual model + create a log file for each date time instant
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename=timestr+'_'+config_data['model']['option']+'_log.logs'
    yaml_filename=timestr+'_'+config_data['model']['option']+'.yaml'

    log_model_subfolder=os.path.join(config_data['output']['log_dir'],config_data['model']['option'])
    if(os.path.exists(log_model_subfolder) is False):
        os.mkdir(log_model_subfolder)
    #create log folder associated with current model
    sub_folder_log=os.path.join(log_model_subfolder,timestr+'_'+config_data['model']['option'])
    if(os.path.exists(sub_folder_log) is False):
        os.mkdir(sub_folder_log)

    #create model folder associated with current model
    model_loc_subfolder=os.path.join(config_data['output']['model_dir'],config_data['model']['option'])
    if(os.path.exists(model_loc_subfolder) is False):
        os.mkdir(model_loc_subfolder)
    
    sub_folder_model=os.path.join(model_loc_subfolder,timestr+'_'+config_data['model']['option'])
    if(os.path.exists(sub_folder_model) is False):
        os.mkdir(sub_folder_model)

    #save the current config in the log_dir 
    yaml_file_name=os.path.join(sub_folder_log,yaml_filename)
    print(yaml_file_name)
    with open(yaml_file_name, "w") as f:
        yaml.dump(config_data, f)


    logger = log(path=sub_folder_log, file=filename)
    logger.info('Starting training')
    logger.info(config_data)


    early_stop_counter=config_data['parameters']['early_stop']
    print('Early stop criteria:%d' %(early_stop_counter))
    early_stop_cnt=0
    train_loss_stats=[]
    val_loss_stats=[]

    val_f1_best=0   
    log_softmax=nn.LogSoftmax(dim=-1)

    ############################## start watching the model using wandb ####################
    wandb.watch(model)

    for epoch in range(1, max_epochs+1): #main outer loop
        train_loss_list=[]
        train_logits=[]
        step=0
        t = time.time()
        target_labels=[]
        pred_labels=[]
        val_loss_list=[]
        for id,(vid_feat,label,lens) in enumerate(tqdm(train_dl)):

            vid_feat=vid_feat.float()
            vid_feat=vid_feat.to(device)
            label = label.type(torch.LongTensor)
            label=label.to(device)
            #print(vid_feat.size())

            #sort wrt lengths
            vid_feat,label,lens = sort_batch(vid_feat,label,lens)

            optim_example.zero_grad()
            logits = model(vid_feat,lens.cpu().numpy())
            
            # Calculate loss
            loss = criterion(logits, label)

            # Back prop.
            loss.backward()
            optim_example.step()
            train_loss_list.append(loss.item())
            target_labels.append(label.cpu())
            train_logits_temp=log_softmax(logits).to('cpu')
            y_pred=torch.max(train_logits_temp, 1)[1]

            #print(label,y_pred)
            train_logits.append(y_pred)
            step=step+1
            
            if(step%150==0):
                logger_step_dict={'Running_Train_loss':mean(train_loss_list)}
                logger.info("Training loss:{:.3f}".format(loss.item()))
                wandb.log(logger_step_dict)


        target_label_np=torch.cat(target_labels).detach().numpy()
        train_predictions = torch.cat(train_logits).detach().numpy()
        
        #train stats
        train_accuracy=accuracy_score(target_label_np,train_predictions)
        f1_score_train=f1_score(target_label_np, train_predictions, average='macro')  
        precision_score_train=precision_score(target_label_np,train_predictions,average='macro')
        recall_score_train=recall_score(target_label_np,train_predictions,average='macro')

        #logger information
        logger.info('epoch: {:d}, time:{:.2f}'.format(epoch, time.time()-t))
        logger.info('\ttrain_loss:{:.3f}, train accuracy:{:.3f}, train f1:{:.3f}'.format(mean(train_loss_list),train_accuracy,f1_score_train))

        #evaluate here 
        logger.info('Evaluating the dataset')
        valid_accuracy,f1_score_val,precision_score_val,recall_score_val,val_loss=gen_validate_score_LSTM_model(val_dl,model,device,criterion)
        logger.info('Validation accuracy:{:.3f},Validation f1:{:.3f}'.format(valid_accuracy,f1_score_val))
        model.train(True)

        #wandb logging
        metrics_dict={'Train_loss':mean(train_loss_list),
            'Train_accuracy':train_accuracy,
            'Train_F1':f1_score_train,
            'Valid_loss':val_loss,
            'Valid_accuracy':valid_accuracy,
            'Valid_F1':f1_score_val,
            'Epoch':epoch}   #add epoch here to later switch the x-axis with epoch rather than actual wandb log

        wandb.log(metrics_dict)

        if(f1_score_val>val_f1_best):
            val_f1_best=f1_score_val
            logger.info('Saving the best model')
            torch.save(model, os.path.join(sub_folder_model,timestr+'_'+config_data['model']['model_type']+'_best_model.pt'))
            early_stop_cnt=0
        else:
            early_stop_cnt=early_stop_cnt+1
        
        if(early_stop_cnt==early_stop_counter):
            print('Validation performance does not improve for %d iterations' %(early_stop_counter))
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', help='Location of configuration data', type=str, required=True)
    args = vars(parser.parse_args())
    config_data=load_config(args['config_file'])
    main(config_data)



