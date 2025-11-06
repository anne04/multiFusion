import os
import sys
import numpy as np
from datetime import datetime 
import time
import random
import argparse
import torch
from multiFusion_train_util import data_to_tensor, get_metadata
from multiEmbFusion import train_multiFusion, test_multiFusion
import pickle
import gzip
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # =========================== must be provided ===============================
    parser.add_argument( '--training_data', type=str, default='Multimodal-Quiz/training_data_multiFusion.pkl', help='Path to training data')
    parser.add_argument( '--model_name', type=str, default="multiFusion_test", help='Provide a model name')
    parser.add_argument( '--fold_count', type=int, default=5, help='Provide the total fold')
    parser.add_argument( '--kfold_info_file', type=str, default="Multimodal-Quiz/data_split_5fold.csv", help='Provide the file name with k fold info')
    #=========================== default is set ======================================
    parser.add_argument( '--num_epoch', type=int, default=5000, help='Number of epochs or iterations for model training')
    parser.add_argument( '--model_path', type=str, default='model/', help='Path to save the model state') # We do not need this for output generation  
    parser.add_argument( '--dropout', type=float, default=0)
    parser.add_argument( '--batch_size', type=int, default=10)
    parser.add_argument( '--lr_rate', type=float, default=0.0001)
    parser.add_argument( '--manual_seed', type=str, default='no', help='Use it for reproducible result')
    parser.add_argument( '--seed', type=int, help='Use it for reproducible result')
    #=========================== optional ======================================
    parser.add_argument( '--load', type=int, default=0, help='Set 1 to load a previously saved model state')  
    parser.add_argument( '--load_model_name', type=str, default='None' , help='Provide the model name that you want to reload')
    #============================================================================
    args = parser.parse_args() 

    args.training_data = args.training_data     
    args.model_path = args.model_path + '/'
    if args.manual_seed == 'yes':
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path) 

    print ('------------------------Model and Training Details--------------------------')
    
    print(args) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    ##########################################################################################
    with gzip.open(args.training_data, 'rb') as fp:  
        patient_vs_modality_vs_image, patient_vs_BCR = pickle.load(fp)

    print('training data load done')


    ##### get the folding set #######
    five_fold_info = pd.read_csv(args.kfold_info_file)
    # prepare a list for holding the k fold info
    fold_vs_test_patient_list = [] 
    for k in range(0, args.fold_count):
        fold_vs_test_patient_list.append([])

    for i in range (0, len(five_fold_info)):
        patient_id = int(five_fold_info['patient_id'][i])
        fold_id = int(five_fold_info['fold'][i])
        # insert into appropriate list
        fold_vs_test_patient_list[fold_id].append(patient_id)

    # get metadata to add padding if slide is missing or row/column mismatch between samples
    metadata_adc, metadata_hbv, metadata_t2w = get_metadata(patient_vs_modality_vs_image)
    ## maximum [C, H, W] for each modality 
    '''
    In [8]: metadata_adc
    Out[8]: [25, 128, 120] --> after first maxpool2d: [25, 64, 60] --> after 2nd maxpool2d [25, 32, 30]   

    In [9]: metadata_hbv
    Out[9]: [25, 128, 120] --> after first maxpool2d: [25, 64, 60] --> after 2nd maxpool2d [25, 32, 30]

    In [10]: metadata_t2w
    Out[10]: [25, 640, 640] --> after first maxpool2d: [25, 320, 320] --> after 2nd maxpool2d [25, 160, 160]   
    '''

    #for k in range(0, args.fold_count):
    k=0
    print('*** Running fold k = %d ***'%k)
    # want to saves model for each fold separately
    args.model_name = args.model_path + args.model_name + '_fold' + str(k) 
    
    test_fold_patient_list = fold_vs_test_patient_list[k]
    training_tensor, validation_tensor, test_tensor = data_to_tensor(metadata_adc, metadata_hbv, metadata_t2w,
                                                                    patient_vs_modality_vs_image, 
                                                                    patient_vs_BCR, 
                                                                    test_fold_patient_list)

    # train the model using train and validation set
    train_multiFusion(args, metadata_adc, metadata_hbv, metadata_t2w,
                        training_tensor, validation_tensor, epoch = args.num_epoch,
                        batch_size = 10, learning_rate=args.lr_rate)

    confusion_matrix = test_multiFusion(args.model_name, test_tensor)





    