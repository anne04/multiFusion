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
import warnings 
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    """
    This is the main function that we use to train the multiFusion model, 
    evaluate it using k-fold cross validation, and save the logs. 
    """
    parser = argparse.ArgumentParser()
    # =========================== input parameters ===============================
    parser.add_argument( '--training_data', type=str, default='Multimodal-Quiz/training_data_multiFusion.pkl', help='Path to training data')
    parser.add_argument( '--model_name', type=str, default="2DmultiFusion_test", help='Provide a model name')
    parser.add_argument('--wandb_project_name', type=str, default='2Dmultimodal_fusion', help='provide a project name for wandb log generation')
    parser.add_argument( '--conv_dimension', type=int, default=2, help='Set to 2 or 3, for running 2D or 3D convolution operation respectively for feature extraction')
    parser.add_argument( '--fold_count', type=int, default=5, help='Provide the total fold')
    parser.add_argument( '--kfold_info_file', type=str, default="Multimodal-Quiz/data_split_5fold.csv", help='Provide the file name with k fold info')
    parser.add_argument( '--model_path', type=str, default='submission_package_models/', help='Path to save the model state') # We do not need this for output generation  
    parser.add_argument( '--output_path', type=str, default='submission_package_models/output/', help='Path to save the final performance reports and loss curves in csv format')
    args = parser.parse_args() 

    args.training_data = args.training_data     
    args.model_path = args.model_path + '/'

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path) 

    
    print(args) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    ##########################################################################################
    with gzip.open(args.training_data, 'rb') as fp:  
        patient_vs_modality_vs_image, patient_vs_timeBCR, patient_vs_weight = pickle.load(fp)

    print('data load done')
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
    print('Image dimensions:')
    print(metadata_adc)
    print(metadata_hbv)
    print(metadata_t2w)
    model_name = args.model_name 

    fold_patient_BCR_timeBCR_prediction = defaultdict(list)
    fold_vs_c_index = defaultdict(list)
    for k in range(0, args.fold_count):
        print('*** Running inference fold k = %d ***'%k)
        test_fold_patient_list = fold_vs_test_patient_list[k]
        if args.conv_dimension == 2:
            no_need, no_need, test_tensor, patient_order = data_to_tensor(metadata_adc, metadata_hbv, metadata_t2w,
                                                                            patient_vs_modality_vs_image, 
                                                                            patient_vs_timeBCR, 
                                                                            patient_vs_weight,
                                                                            test_fold_patient_list,
                                                                            [],
                                                                            validation_percent = 25
                                                                            )
        elif args.conv_dimension == 3:
            no_need, no_need, test_tensor, patient_order = data_to_3Dtensor(metadata_adc, metadata_hbv, metadata_t2w,
                                                                            patient_vs_modality_vs_image, 
                                                                            patient_vs_timeBCR, 
                                                                            patient_vs_weight,
                                                                            test_fold_patient_list,
                                                                            [],
                                                                            validation_percent = 25
                                                                            )

        # load model for each fold separately        
        args.model_name = args.model_path + model_name + '_fold' + str(k) 
        # send the model name to load it for evaluating test data
        c_index, prediction_order = test_multiFusion(args.model_name, test_tensor, k)
        # patient_order and prediction_order has same order
        for i in range(0, len(prediction_order)):
            patient_id = patient_order[i]
            fold_patient_BCR_timeBCR_prediction['fold'].append(k)
            fold_patient_BCR_timeBCR_prediction['patient_id'].append(patient_id)
            fold_patient_BCR_timeBCR_prediction['original_time_of_BCR (100 means BCR=0)'].append(patient_vs_timeBCR[patient_id])
            fold_patient_BCR_timeBCR_prediction['predicted_BCR'].append(prediction_order[i])


        fold_vs_c_index['fold'].append(k)
        fold_vs_c_index['c-index'].append(c_index)


    ###########
    case_by_case_report = pd.DataFrame(fold_patient_BCR_timeBCR_prediction)
    case_by_case_report.to_csv(args.output_path + '/'+ model_name +'_case_by_case_report.csv', index=False)
    print('Saved at: ' + args.output_path + '/'+ model_name + '_case_by_case_report.csv')

    fold_vs_c_index_report = pd.DataFrame(fold_vs_c_index)
    fold_vs_c_index_report.to_csv(args.output_path + '/'+ model_name + '_fold_vs_c_index_report.csv', index=False)
    print('Saved at: ' + args.output_path + '/'+ model_name + '_fold_vs_c_index_report.csv')

    print('All done.')
