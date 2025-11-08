import os
import sys
import numpy as np
from datetime import datetime 
import time
import random
import argparse
import torch
from multiFusion_train_util import data_to_tensor, data_to_3Dtensor, get_metadata
from multiEmbFusion import train_multiFusion, test_multiFusion
import pickle
import gzip
import pandas as pd
from collections import defaultdict

if __name__ == "__main__":
    """
    This is the main function that we use to train the multiFusion model, 
    evaluate it using k-fold cross validation, and save the logs. 
    """
    parser = argparse.ArgumentParser()
    # =========================== must be provided ===============================
    parser.add_argument( '--training_data', type=str, default='Multimodal-Quiz/training_data_multiFusion.pkl', help='Path to training data')
    parser.add_argument( '--model_name', type=str, default="3DmultiFusion_test", help='Provide a model name')
    parser.add_argument( '--fold_count', type=int, default=5, help='Provide the total fold')
    parser.add_argument( '--kfold_info_file', type=str, default="Multimodal-Quiz/data_split_5fold.csv", help='Provide the file name with k fold info')
    #=========================== default is set ======================================
    parser.add_argument( '--conv_dimension', type=int, default=3)
    parser.add_argument( '--num_epoch', type=int, default=1000, help='Number of epochs or iterations for model training')
    parser.add_argument( '--model_path', type=str, default='model/', help='Path to save the model state') # We do not need this for output generation  
    parser.add_argument( '--output_path', type=str, default='output/', help='Path to save the model state')
    parser.add_argument( '--dropout', type=float, default=0)
    parser.add_argument( '--batch_size', type=int, default=10)
    parser.add_argument( '--lr_rate', type=float, default=0.001)
    parser.add_argument( '--manual_seed', type=str, default='no', help='Use it for reproducible result')
    parser.add_argument( '--seed', type=int, help='Use it for reproducible result')
    parser.add_argument('--wandb_project_name', type=str, default='3Dmultimodal_fusion')
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

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path) 

    print ('------------------------Model and Training Details--------------------------')
    
    print(args) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    ##########################################################################################
    with gzip.open(args.training_data, 'rb') as fp:  
        patient_vs_modality_vs_image, patient_vs_timeBCR = pickle.load(fp)

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
    fold_patient_BCR_timeBCR_prediction = defaultdict(list)
    fold_vs_c_index = defaultdict(list)
    print_model_flag = 1
    model_name = args.model_name 
    for k in range(0, args.fold_count):
    #k=0
        print('*** Running fold k = %d ***'%k)
        test_fold_patient_list = fold_vs_test_patient_list[k]
        if args.conv_dimension == 2:
            training_tensor, validation_tensor, test_tensor, patient_order = data_to_tensor(metadata_adc, metadata_hbv, metadata_t2w,
                                                                            patient_vs_modality_vs_image, 
                                                                            patient_vs_timeBCR, 
                                                                            test_fold_patient_list
                                                                            )
        elif args.conv_dimension == 3:
            training_tensor, validation_tensor, test_tensor, patient_order = data_to_3Dtensor(metadata_adc, metadata_hbv, metadata_t2w,
                                                                            patient_vs_modality_vs_image, 
                                                                            patient_vs_timeBCR, 
                                                                            test_fold_patient_list
                                                                            )

        # train the model using train and validation set
        # want to saves model for each fold separately
        
        args.model_name = args.model_path + model_name + '_fold' + str(k) 
        wandb_project_name = args.wandb_project_name 
        train_multiFusion(args, metadata_adc, metadata_hbv, metadata_t2w,
                            training_tensor, validation_tensor, epoch = args.num_epoch,
                            batch_size = 10, learning_rate=args.lr_rate, print_model_flag = print_model_flag, 
                            wandb_project_name=wandb_project_name, fold=k)

        print_model_flag = 0
        print('*** testing now ***')
        c_index, prediction_order = test_multiFusion(args.model_name, test_tensor, k)
        # patient_order and prediction_order has same order
        for i in range(0, len(prediction_order)):
            patient_id = patient_order[i]
            fold_patient_BCR_timeBCR_prediction['fold'].append(k)
            fold_patient_BCR_timeBCR_prediction['patient_id'].append(patient_id)
            #fold_patient_BCR_timeBCR_prediction['BCR(preprocessed)'].append(patient_vs_BCR[patient_id])
            fold_patient_BCR_timeBCR_prediction['original_time_of_BCR (100 means BCR=0)'].append(patient_vs_timeBCR[patient_id])
            fold_patient_BCR_timeBCR_prediction['predicted_BCR'].append(prediction_order[i])


        fold_vs_c_index['fold'].append(k)
        fold_vs_c_index['c-index'].append(c_index)


    ###########
    case_by_case_report = pd.DataFrame(fold_patient_BCR_timeBCR_prediction)
    case_by_case_report.to_csv(args.output_path + '/'+ args.model_name +'_case_by_case_report.csv', index=False)

    fold_vs_c_index_report = pd.DataFrame(fold_vs_c_index)
    fold_vs_c_index_report.to_csv(args.output_path + '/'+ args.model_name +'_fold_vs_c_index_report.csv', index=False)

    



      
