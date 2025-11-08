import SimpleITK as sitk
import numpy as np
from collections import defaultdict
from pathlib import Path
import json
import pandas as pd
import pickle
import gzip
import argparse


# USER INPUT
#path_dataset_mpMRI = 'Multimodal-Quiz/radiology/mpMRI/'
#path_dataset_prost_mask_t2w = 'Multimodal-Quiz/radiology/prostate_mask_t2w/'
#path_dataset_clinical = 'Multimodal-Quiz/clinical_data/'
# prepare the dataset

########################### radiology ################################################


def data_preprocess_multimodal(args):
    # This function takes user input through args, preprocess data, and save in required format.
    # Two files are saved: 
    # training_data_multiFusion.pkl --> model training
    # patient_vs_clinical_data.csv --> just for metadata visualization

    patient_vs_modality_vs_image = defaultdict(dict)
    modality_vs_valueRange = defaultdict(list)
    # patient_vs_modality_vs_image[patient_id][modality_type] = image array

    # 
    patient_id_list = [d.name for d in Path(path_dataset_mpMRI).iterdir() if d.is_dir()]
    print(patient_id_list)

    for patient_id in patient_id_list:
        print('patient id: ' + patient_id + ' reading...')
        for modality in ['adc', 'hbv']:
            file_path = path_dataset_mpMRI + '/' + patient_id + '/' + patient_id + '_0001_'+modality+'.mha'
            image = sitk.ReadImage(file_path)
            image_array = sitk.GetArrayFromImage(image)
            #print(image_array.shape)
            patient_vs_modality_vs_image[patient_id][modality] = image_array
            modality_vs_valueRange[modality].append(np.min(image_array))
            modality_vs_valueRange[modality].append(np.max(image_array))
            


        modality = 't2w'
        file_path = path_dataset_mpMRI + '/' + patient_id + '/' + patient_id + '_0001_'+modality+'.mha'
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)
        #print(image_array.shape)
        # this one needs a mask
        mask_file_path = path_dataset_prost_mask_t2w + '/' + patient_id + '_0001_' + 'mask' + '.mha'
        mask_image = sitk.ReadImage(mask_file_path)
        mask_image_array = sitk.GetArrayFromImage(mask_image)
        # print(mask_image_array.shape)
        # multiply image_array with mask 
        masked_image = image_array * mask_image_array
        # print('1 values item in mask is ' + str(np.sum(mask_image_array))) # for debug
        # print('1 values item in t2w masked image is ' + str(np.count_nonzero(masked_image))) # for debug
        patient_vs_modality_vs_image[patient_id][modality] = masked_image
        modality_vs_valueRange[modality].append(np.min(masked_image))
        modality_vs_valueRange[modality].append(np.max(masked_image))


    # print to see the min max range for each modality
    '''
    for modality in modality_vs_valueRange:
        print('for modality '+ modality + ' intensity ranges from ' + str(np.min(modality_vs_valueRange[modality]))
            + ' to ' + str(np.max(modality_vs_valueRange[modality])))

    '''
    # how many entries in prostate_tw2 are 1 for each patient
    '''
    for patient_id in patient_vs_modality_vs_image:
        count_one = np.sum(patient_vs_modality_vs_image[patient_id]['prost_t2w'])
        print(patient_id + ' has ' + str(count_one) + ' entries as 1')
    '''

    ######################### clinical ################################################
    # get the max number of features first
    clinical_features = dict()
    for patient_id in patient_id_list:
        json_file_path = path_dataset_clinical + '/' + patient_id + '.json'
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        for key in data:
            clinical_features[key] = ''

    clinical_features = list(clinical_features.keys())

    # Now get the clinical value
    patient_vs_clinical_data = defaultdict(list) # patient_vs_clinical_data[patient_id] = vector having clinical info 
    patient_vs_feature_vs_value = defaultdict(dict)
    for patient_id in patient_id_list:
        json_file_path = path_dataset_clinical + '/' + patient_id + '.json'
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        #print('patient id ' + patient_id + 'has keys ' + str(len(list(data.keys()))))
        patient_vs_clinical_data['patient_id'].append(patient_id)
        for key in clinical_features:
            if key not in data: # handle missing value
                patient_vs_clinical_data[key].append(-1)
                patient_vs_feature_vs_value[patient_id][key] = -1
            else:
                patient_vs_clinical_data[key].append(data[key])
                patient_vs_feature_vs_value[patient_id][key] = data[key]

                
    # get patient vs time to BCR - for model training 
    patient_vs_timeBCR = dict()
    for patient_id in patient_id_list:
        if int(float(patient_vs_feature_vs_value[patient_id]['BCR']))==0: # BCR did not happen
            patient_vs_timeBCR[patient_id] = 100.0 # some high value to indicate long time to BCR

        else: # record the actual time of BCR
            patient_vs_timeBCR[patient_id] = float(patient_vs_feature_vs_value[patient_id]['time_to_follow-up/BCR'])


    # convert it to csv file for future reference
    df = pd.DataFrame(patient_vs_clinical_data)
    df.to_csv(output_path + '/patient_vs_clinical_data.csv', index=False)
    print('Saved at: '+ output_path + '/patient_vs_clinical_data.csv')

    ## now pack the training dataset and save as pickle object
    with gzip.open(output_path + '/training_data_multiFusion.pkl', 'wb') as fp:  
        pickle.dump([patient_vs_modality_vs_image, patient_vs_timeBCR], fp)

    print('Saved at: ' + output_path + '/training_data_multiFusion.pkl')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # =========================== must be provided ===============================
    parser.add_argument( '--path_dataset_mpMRI', type=str, default='Multimodal-Quiz/radiology/mpMRI/', help='Path to training data')
    parser.add_argument( '--path_dataset_prost_mask_t2w', type=str, default="'Multimodal-Quiz/radiology/prostate_mask_t2w/'", help='Provide a model name')
    parser.add_argument( '--path_dataset_clinical', type=str, default='Multimodal-Quiz/clinical_data/', help='Provide the total fold')
    parser.add_argument( '--output_path', type=str, default='Multimodal-Quiz/', help='Path to save the model state')
    #============================================================================
    args = parser.parse_args() 

    data_preprocess_multimodal(args)


    exit(0)
