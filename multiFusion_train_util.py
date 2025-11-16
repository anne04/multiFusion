import torch
import numpy as np
from scipy.ndimage import rotate

def shuffle_data(
    training_set
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function takes input the training data in tensor format, shuffles it and then 
    separates into multiple branches (adc, hbv, t2w, target) to be passed through the multi-modal deep learning model.  
    Args:
        training_set (list):
        This is a list of adc, hbv, and t2w feature tensors, and corresponding target value tensors. 
        Each of adc, hbv, and t2w tensors has dimension [patient_count, slide_count, slide_height, slide_width].
        Target value tensor has dimension [patient_count, 1]. 
    Returns:
        Separate tensors of ADC, HBV, T2W, and Target values 

    """
    patient_vs_adc_train = training_set[0]
    patient_vs_hbv_train = training_set[1]
    patient_vs_t2w_train = training_set[2]
    patient_vs_prediction_train = training_set[3]
    patient_vs_weight_train = training_set[4]

    # Generate random permutation of row indices
    sample_count = patient_vs_adc_train.shape[0]
    #prediction_column = training_set.size(1)-1
    row_perm = torch.randperm(sample_count)

    # Shuffle the rows using advanced indexing # 
    patient_vs_adc_train = patient_vs_adc_train[row_perm]
    patient_vs_hbv_train = patient_vs_hbv_train[row_perm] 
    patient_vs_t2w_train = patient_vs_t2w_train[row_perm]
    patient_vs_prediction_train = patient_vs_prediction_train[row_perm]
    patient_vs_weight_train = patient_vs_weight_train[row_perm]

    # Now augment: Org, random Gaussian Noise, random rotation - 3 times
    training_data_count = sample_count*5
    patient_vs_adc_tensors_train = np.zeros((training_data_count, patient_vs_adc_train[0].shape[0], patient_vs_adc_train[0].shape[1], patient_vs_adc_train[0].shape[2]))
    patient_vs_hbv_tensors_train = np.zeros((training_data_count, patient_vs_hbv_train[0].shape[0], patient_vs_hbv_train[0].shape[1], patient_vs_hbv_train[0].shape[2]))
    patient_vs_t2w_tensors_train = np.zeros((training_data_count, patient_vs_t2w_train[0].shape[0], patient_vs_t2w_train[0].shape[1], patient_vs_t2w_train[0].shape[2]))
    patient_vs_prediction_tensors_train = np.zeros((training_data_count, 1))
    patient_vs_weight_tensors_train = np.zeros((training_data_count, 1))

    i_train = 0
    for indx in range (0, sample_count):

        image_array_adc = patient_vs_adc_train[indx]
        image_array_hbv = patient_vs_hbv_train[indx]
        image_array_t2w = patient_vs_t2w_train[indx]

        prediction = patient_vs_prediction_train[indx]
        sample_weight = patient_vs_weight_train[indx]

        ############# org ##############
        patient_vs_adc_tensors_train[i_train][0:image_array_adc.shape[0],0:image_array_adc.shape[1],0:image_array_adc.shape[2]] = image_array_adc
        patient_vs_hbv_tensors_train[i_train][0:image_array_hbv.shape[0],0:image_array_hbv.shape[1],0:image_array_hbv.shape[2]] = image_array_hbv
        patient_vs_t2w_tensors_train[i_train][0:image_array_t2w.shape[0],0:image_array_t2w.shape[1],0:image_array_t2w.shape[2]] = image_array_t2w
        patient_vs_prediction_tensors_train[i_train][0] = prediction
        patient_vs_weight_tensors_train[i_train][0] = sample_weight
        i_train = i_train + 1

        ######### rotate along HxW plane ###############
        angle = np.random.randint(low=-15, high=+15)
        image_array_adc_temp = rotate(image_array_adc, angle=angle, axes=(1, 2), reshape=False)
        image_array_hbv_temp = rotate(image_array_hbv, angle=angle, axes=(1, 2), reshape=False)
        image_array_t2w_temp = rotate(image_array_t2w, angle=angle, axes=(1, 2), reshape=False)

        patient_vs_adc_tensors_train[i_train][0:image_array_adc.shape[0],0:image_array_adc.shape[1],0:image_array_adc.shape[2]] = image_array_adc_temp
        patient_vs_hbv_tensors_train[i_train][0:image_array_hbv.shape[0],0:image_array_hbv.shape[1],0:image_array_hbv.shape[2]] = image_array_hbv_temp
        patient_vs_t2w_tensors_train[i_train][0:image_array_t2w.shape[0],0:image_array_t2w.shape[1],0:image_array_t2w.shape[2]] = image_array_t2w_temp
        patient_vs_prediction_tensors_train[i_train][0] = prediction
        patient_vs_weight_tensors_train[i_train][0] = sample_weight
        i_train = i_train + 1

        # gaussian Noise ################
        image_array_adc_temp= image_array_adc + np.random.normal(0, 0.01, image_array_adc.shape)
        image_array_hbv_temp = image_array_hbv + np.random.normal(0, 0.01, image_array_hbv.shape)
        image_array_t2w_temp = image_array_t2w + np.random.normal(0, 0.01, image_array_t2w.shape)

        patient_vs_adc_tensors_train[i_train][0:image_array_adc.shape[0],0:image_array_adc.shape[1],0:image_array_adc.shape[2]] = scale_image(image_array_adc_temp)
        patient_vs_hbv_tensors_train[i_train][0:image_array_hbv.shape[0],0:image_array_hbv.shape[1],0:image_array_hbv.shape[2]] = scale_image(image_array_hbv_temp)
        patient_vs_t2w_tensors_train[i_train][0:image_array_t2w.shape[0],0:image_array_t2w.shape[1],0:image_array_t2w.shape[2]] = scale_image(image_array_t2w_temp)
        patient_vs_prediction_tensors_train[i_train][0] = prediction
        patient_vs_weight_tensors_train[i_train][0] = sample_weight
        i_train = i_train + 1
        ######### rotate along HxW plane ###############
        angle = np.random.randint(low=-15, high=+15)
        image_array_adc_temp = rotate(image_array_adc, angle=angle, axes=(1, 2), reshape=False)
        image_array_hbv_temp = rotate(image_array_hbv, angle=angle, axes=(1, 2), reshape=False)
        image_array_t2w_temp = rotate(image_array_t2w, angle=angle, axes=(1, 2), reshape=False)

        patient_vs_adc_tensors_train[i_train][0:image_array_adc.shape[0],0:image_array_adc.shape[1],0:image_array_adc.shape[2]] = image_array_adc_temp
        patient_vs_hbv_tensors_train[i_train][0:image_array_hbv.shape[0],0:image_array_hbv.shape[1],0:image_array_hbv.shape[2]] = image_array_hbv_temp
        patient_vs_t2w_tensors_train[i_train][0:image_array_t2w.shape[0],0:image_array_t2w.shape[1],0:image_array_t2w.shape[2]] = image_array_t2w_temp
        patient_vs_prediction_tensors_train[i_train][0] = prediction
        patient_vs_weight_tensors_train[i_train][0] = sample_weight
        i_train = i_train + 1

        # gaussian Noise ################
        image_array_adc_temp= image_array_adc + np.random.normal(0, 0.01, image_array_adc.shape)
        image_array_hbv_temp = image_array_hbv + np.random.normal(0, 0.01, image_array_hbv.shape)
        image_array_t2w_temp = image_array_t2w + np.random.normal(0, 0.01, image_array_t2w.shape)

        patient_vs_adc_tensors_train[i_train][0:image_array_adc.shape[0],0:image_array_adc.shape[1],0:image_array_adc.shape[2]] = scale_image(image_array_adc_temp)
        patient_vs_hbv_tensors_train[i_train][0:image_array_hbv.shape[0],0:image_array_hbv.shape[1],0:image_array_hbv.shape[2]] = scale_image(image_array_hbv_temp)
        patient_vs_t2w_tensors_train[i_train][0:image_array_t2w.shape[0],0:image_array_t2w.shape[1],0:image_array_t2w.shape[2]] = scale_image(image_array_t2w_temp)
        patient_vs_prediction_tensors_train[i_train][0] = prediction
        patient_vs_weight_tensors_train[i_train][0] = sample_weight
        i_train = i_train + 1


    # now convert to tensor

    patient_vs_adc_tensors_train = torch.tensor(patient_vs_adc_tensors_train, dtype=torch.float)
    patient_vs_hbv_tensors_train = torch.tensor(patient_vs_hbv_tensors_train, dtype=torch.float)
    patient_vs_t2w_tensors_train = torch.tensor(patient_vs_t2w_tensors_train, dtype=torch.float)
    patient_vs_prediction_tensors_train = torch.tensor(patient_vs_prediction_tensors_train, dtype=torch.float)
    patient_vs_weight_tensors_train = torch.tensor(patient_vs_weight_tensors_train, dtype=torch.float)


    return patient_vs_adc_tensors_train, patient_vs_hbv_tensors_train, patient_vs_t2w_tensors_train, patient_vs_prediction_tensors_train, patient_vs_weight_tensors_train



def scale_image(img):
    """
    This function scales the intensities of input image into 0-255 range 
    and returns it.
    """

    img_min = np.min(img)
    img_max = np.max(img)

    if img_max - img_min == 0:
        scaled_img = 0 * img
                            
    else:
        scaled_img = 255 * (img - img_min) / (img_max - img_min)
    #scaled_img = scaled_img.astype(np.uint8)
    return scaled_img



def get_metadata(patient_vs_modality_vs_image):
    """
    This function takes the preprocessed training data in dictionary format and returns the metadata
    info for scans coming from each modality: [Maximum channel, Maximum Height, Maximum Width]    
    """
    ### need them so that we can pad 0 to those missing slides or smaller slide size #####
    max_H_adc = []
    max_W_adc = []
    max_C_adc = []

    max_H_hbv = []
    max_W_hbv = []
    max_C_hbv = []

    max_H_t2w = []
    max_W_t2w = []
    max_C_t2w = []

    for patient_id in patient_vs_modality_vs_image:
        
        max_H_adc.append(patient_vs_modality_vs_image[patient_id]['adc'].shape[1])
        max_W_adc.append(patient_vs_modality_vs_image[patient_id]['adc'].shape[2])
        max_C_adc.append(patient_vs_modality_vs_image[patient_id]['adc'].shape[0])

        max_H_hbv.append(patient_vs_modality_vs_image[patient_id]['hbv'].shape[1])
        max_W_hbv.append(patient_vs_modality_vs_image[patient_id]['hbv'].shape[2])
        max_C_hbv.append(patient_vs_modality_vs_image[patient_id]['hbv'].shape[0])


        max_H_t2w.append(patient_vs_modality_vs_image[patient_id]['t2w'].shape[1])
        max_W_t2w.append(patient_vs_modality_vs_image[patient_id]['t2w'].shape[2])
        max_C_t2w.append(patient_vs_modality_vs_image[patient_id]['t2w'].shape[0])



    max_H_adc = np.max(max_H_adc)
    max_W_adc = np.max(max_W_adc)
    max_C_adc = np.max(max_C_adc)

    max_H_hbv = np.max(max_H_hbv)
    max_W_hbv = np.max(max_W_hbv)
    max_C_hbv = np.max(max_C_hbv)

    max_H_t2w = np.max(max_H_t2w)
    max_W_t2w = np.max(max_W_t2w)
    max_C_t2w = np.max(max_C_t2w)

    metadata_adc = [max_C_adc, max_H_adc, max_W_adc]
    metadata_hbv = [max_C_hbv, max_H_hbv, max_W_hbv]
    metadata_t2w = [max_C_t2w, max_H_t2w, max_W_t2w]

    return metadata_adc, metadata_hbv, metadata_t2w

def data_to_tensor(
    metadata_adc,
    metadata_hbv,
    metadata_t2w,
    patient_vs_modality_vs_image, 
    patient_vs_timeBCR,
    patient_vs_weight,
    test_fold_patient_list,
    validation_fold_patient_list,
    validation_percent = -1,
    ):
    """
    This model takes the training data in raw format and convert them to tensors.

    Args:
        metadata_adc (list): List of channel, height, width information for ADC slides,
        metadata_hbv (list): List of channel, height, width information for HBV slides,
        metadata_t2w (list): List of channel, height, width information for T2W slides,
        patient_vs_modality_vs_image (dictionary): It has image array per each modality for each patient, 
        patient_vs_timeBCR (dictionary): It has time to BCR information for each patient,
        test_fold_patient_list (list): It is a list for testing patient for current fold,
        validation_percent (int): Percentage of training data (four folds) to keep for validation. 

    Return:
        tensor format of training_set, validation_set, testing_set. It also returns the patient_order in 
        testing set which is used for logging case by case report for each fold.

    """


    ####### initialze placeholder arrays that will be converter to tensor #######

    testing_data_count = len(test_fold_patient_list)
    validation_data_count = len(validation_fold_patient_list)
    if len(validation_fold_patient_list)!=0:
        training_data_count = len(list(patient_vs_modality_vs_image.keys())) - testing_data_count - validation_data_count
    else:
        training_data_count = len(list(patient_vs_modality_vs_image.keys())) - testing_data_count





    patient_vs_adc_tensors_train = np.zeros((training_data_count, metadata_adc[0], metadata_adc[1], metadata_adc[2]))
    patient_vs_adc_tensors_test = np.zeros((testing_data_count, metadata_adc[0], metadata_adc[1], metadata_adc[2]))
    patient_vs_adc_tensors_validation = np.zeros((validation_data_count, metadata_adc[0], metadata_adc[1], metadata_adc[2]))


    patient_vs_hbv_tensors_train = np.zeros((training_data_count, metadata_hbv[0], metadata_hbv[1], metadata_hbv[2]))
    patient_vs_hbv_tensors_test = np.zeros((testing_data_count, metadata_hbv[0], metadata_hbv[1], metadata_hbv[2]))
    patient_vs_hbv_tensors_validation = np.zeros((validation_data_count, metadata_hbv[0], metadata_hbv[1], metadata_hbv[2]))

    patient_vs_t2w_tensors_train = np.zeros((training_data_count, metadata_t2w[0], metadata_t2w[1], metadata_t2w[2]))
    patient_vs_t2w_tensors_test = np.zeros((testing_data_count, metadata_t2w[0], metadata_t2w[1], metadata_t2w[2]))
    patient_vs_t2w_tensors_validation = np.zeros((validation_data_count, metadata_t2w[0], metadata_t2w[1], metadata_t2w[2]))

    patient_vs_prediction_tensors_train = np.zeros((training_data_count, 1))
    patient_vs_prediction_tensors_test = np.zeros((testing_data_count,1))
    patient_vs_prediction_tensors_validation = np.zeros((validation_data_count,1))

    patient_vs_weight_tensors_train = np.zeros((training_data_count, 1))

    i_test = 0
    i_train = 0
    i_val = 0
    patient_order = []
    for patient_id in patient_vs_modality_vs_image:

        image_array_adc = patient_vs_modality_vs_image[patient_id]['adc']
        image_array_hbv = patient_vs_modality_vs_image[patient_id]['hbv']
        image_array_t2w = patient_vs_modality_vs_image[patient_id]['t2w']

        prediction = patient_vs_timeBCR[patient_id]
        sample_weight = patient_vs_weight[patient_id]

        if int(patient_id) in test_fold_patient_list:
            patient_vs_adc_tensors_test[i_test][0:image_array_adc.shape[0],0:image_array_adc.shape[1],0:image_array_adc.shape[2]] = scale_image(image_array_adc)
            patient_vs_hbv_tensors_test[i_test][0:image_array_hbv.shape[0],0:image_array_hbv.shape[1],0:image_array_hbv.shape[2]] = scale_image(image_array_hbv)
            patient_vs_t2w_tensors_test[i_test][0:image_array_t2w.shape[0],0:image_array_t2w.shape[1],0:image_array_t2w.shape[2]] = scale_image(image_array_t2w)
            patient_vs_prediction_tensors_test[i_test][0] = prediction
            i_test = i_test + 1
            patient_order.append(patient_id)

        elif int(patient_id) in validation_fold_patient_list:
            patient_vs_adc_tensors_validation[i_val][0:image_array_adc.shape[0],0:image_array_adc.shape[1],0:image_array_adc.shape[2]] = scale_image(image_array_adc)
            patient_vs_hbv_tensors_validation[i_val][0:image_array_hbv.shape[0],0:image_array_hbv.shape[1],0:image_array_hbv.shape[2]] = scale_image(image_array_hbv)
            patient_vs_t2w_tensors_validation[i_val][0:image_array_t2w.shape[0],0:image_array_t2w.shape[1],0:image_array_t2w.shape[2]] = scale_image(image_array_t2w)
            patient_vs_prediction_tensors_validation[i_val][0] = prediction
            i_val = i_val + 1

        else:
            patient_vs_adc_tensors_train[i_train][0:image_array_adc.shape[0],0:image_array_adc.shape[1],0:image_array_adc.shape[2]] = scale_image(image_array_adc)
            patient_vs_hbv_tensors_train[i_train][0:image_array_hbv.shape[0],0:image_array_hbv.shape[1],0:image_array_hbv.shape[2]] = scale_image(image_array_hbv)
            patient_vs_t2w_tensors_train[i_train][0:image_array_t2w.shape[0],0:image_array_t2w.shape[1],0:image_array_t2w.shape[2]] = scale_image(image_array_t2w)
            patient_vs_prediction_tensors_train[i_train][0] = prediction
            patient_vs_weight_tensors_train[i_train][0] = sample_weight
            i_train = i_train + 1

    # take 20% of train for validation ########################################################
    
    if validation_percent != -1 :
        sample_count = training_data_count
        row_perm = torch.randperm(sample_count)
        patient_vs_adc_tensors_train = patient_vs_adc_tensors_train[row_perm]
        patient_vs_hbv_tensors_train = patient_vs_hbv_tensors_train[row_perm]
        patient_vs_t2w_tensors_train = patient_vs_t2w_tensors_train[row_perm]
        patient_vs_prediction_tensors_train = patient_vs_prediction_tensors_train[row_perm]
        patient_vs_weight_tensors_train = patient_vs_weight_tensors_train[row_perm]

        validation_count = (training_data_count*validation_percent) // 100
        patient_vs_adc_tensors_validation = patient_vs_adc_tensors_train[0:validation_count]
        patient_vs_adc_tensors_train = patient_vs_adc_tensors_train[validation_count:]
    
        patient_vs_hbv_tensors_validation = patient_vs_hbv_tensors_train[0:validation_count]
        patient_vs_hbv_tensors_train = patient_vs_hbv_tensors_train[validation_count:]

        patient_vs_t2w_tensors_validation = patient_vs_t2w_tensors_train[0:validation_count]
        patient_vs_t2w_tensors_train = patient_vs_t2w_tensors_train[validation_count:]

        patient_vs_prediction_tensors_validation = patient_vs_prediction_tensors_train[0:validation_count]
        patient_vs_prediction_tensors_train = patient_vs_prediction_tensors_train[validation_count:]

        #patient_vs_weight_tensors_validation = patient_vs_weight_tensors_train[0:validation_count]
        patient_vs_weight_tensors_train = patient_vs_weight_tensors_train[validation_count:]    
    
    

    ############################################################################################
    # convert to tensor
    #patient_vs_adc_tensors_train = torch.tensor(patient_vs_adc_tensors_train, dtype=torch.float)
    patient_vs_adc_tensors_validation = torch.tensor(patient_vs_adc_tensors_validation, dtype=torch.float)
    patient_vs_adc_tensors_test = torch.tensor(patient_vs_adc_tensors_test, dtype=torch.float)

    #patient_vs_hbv_tensors_train = torch.tensor(patient_vs_hbv_tensors_train, dtype=torch.float)
    patient_vs_hbv_tensors_validation = torch.tensor(patient_vs_hbv_tensors_validation, dtype=torch.float)
    patient_vs_hbv_tensors_test = torch.tensor(patient_vs_hbv_tensors_test, dtype=torch.float)

    #patient_vs_t2w_tensors_train = torch.tensor(patient_vs_t2w_tensors_train, dtype=torch.float)
    patient_vs_t2w_tensors_validation = torch.tensor(patient_vs_t2w_tensors_validation, dtype=torch.float)
    patient_vs_t2w_tensors_test = torch.tensor(patient_vs_t2w_tensors_test, dtype=torch.float)

    #patient_vs_prediction_tensors_train = torch.tensor(patient_vs_prediction_tensors_train, dtype=torch.float)
    patient_vs_prediction_tensors_validation = torch.tensor(patient_vs_prediction_tensors_validation, dtype=torch.float)
    patient_vs_prediction_tensors_test = torch.tensor(patient_vs_prediction_tensors_test, dtype=torch.float)


    #patient_vs_weight_tensors_train = torch.tensor(patient_vs_weight_tensors_train, dtype=torch.float)
    #patient_vs_weight_tensors_validation = torch.tensor(patient_vs_weight_tensors_validation, dtype=torch.float)


    training_set = [patient_vs_adc_tensors_train, patient_vs_hbv_tensors_train, patient_vs_t2w_tensors_train, patient_vs_prediction_tensors_train, patient_vs_weight_tensors_train]
    testing_set = [patient_vs_adc_tensors_test, patient_vs_hbv_tensors_test, patient_vs_t2w_tensors_test, patient_vs_prediction_tensors_test]
    validation_set = [patient_vs_adc_tensors_validation, patient_vs_hbv_tensors_validation, patient_vs_t2w_tensors_validation, patient_vs_prediction_tensors_validation]
    ############################################################################################

    return training_set, validation_set, testing_set, patient_order

def shuffle_3Ddata(
    training_set
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function takes input the training data in tensor format, shuffles it and then 
    separates into multiple branches (adc, hbv, t2w, target) to be passed through the multi-modal deep learning model.  
    Args:
        training_set (list):
        This is a list of adc, hbv, and t2w feature tensors, and corresponding target value tensors. 
        Each of adc, hbv, and t2w tensors has dimension [patient_count, slide_count, slide_height, slide_width].
        Target value tensor has dimension [patient_count, 1]. 
    Returns:
        Separate tensors of ADC, HBV, T2W, and Target values 

    """
    patient_vs_adc_train = training_set[0]
    patient_vs_hbv_train = training_set[1]
    patient_vs_t2w_train = training_set[2]
    patient_vs_prediction_train = training_set[3]
    patient_vs_weight_train = training_set[4]

    # Generate random permutation of row indices
    sample_count = patient_vs_adc_train.shape[0]
    #prediction_column = training_set.size(1)-1
    row_perm = torch.randperm(sample_count)

    # Shuffle the rows using advanced indexing # 
    patient_vs_adc_train = patient_vs_adc_train[row_perm]
    patient_vs_hbv_train = patient_vs_hbv_train[row_perm] 
    patient_vs_t2w_train = patient_vs_t2w_train[row_perm]
    patient_vs_prediction_train = patient_vs_prediction_train[row_perm]
    patient_vs_weight_train = patient_vs_weight_train[row_perm]

    # Now augment: Org, random Gaussian Noise, random rotation - 3 times
    training_data_count = sample_count*3
    patient_vs_adc_tensors_train = np.zeros((training_data_count, 1, patient_vs_adc_train[0].shape[1], patient_vs_adc_train[0].shape[2], patient_vs_adc_train[0].shape[3]))
    patient_vs_hbv_tensors_train = np.zeros((training_data_count, 1, patient_vs_hbv_train[0].shape[1], patient_vs_hbv_train[0].shape[2], patient_vs_hbv_train[0].shape[3]))
    patient_vs_t2w_tensors_train = np.zeros((training_data_count, 1, patient_vs_t2w_train[0].shape[1], patient_vs_t2w_train[0].shape[2], patient_vs_t2w_train[0].shape[3]))
    patient_vs_prediction_tensors_train = np.zeros((training_data_count, 1))
    patient_vs_weight_tensors_train = np.zeros((training_data_count, 1))

    i_train = 0
    for indx in range (0, sample_count):

        image_array_adc = patient_vs_adc_train[indx]
        image_array_hbv = patient_vs_hbv_train[indx]
        image_array_t2w = patient_vs_t2w_train[indx]

        prediction = patient_vs_prediction_train[indx]
        sample_weight = patient_vs_weight_train[indx]

        ############# org ##############
        patient_vs_adc_tensors_train[i_train][0, 0:image_array_adc.shape[1],0:image_array_adc.shape[2],0:image_array_adc.shape[3]] = image_array_adc
        patient_vs_hbv_tensors_train[i_train][0, 0:image_array_hbv.shape[1],0:image_array_hbv.shape[2],0:image_array_hbv.shape[3]] = image_array_hbv
        patient_vs_t2w_tensors_train[i_train][0, 0:image_array_t2w.shape[1],0:image_array_t2w.shape[2],0:image_array_t2w.shape[3]] = image_array_t2w
        patient_vs_prediction_tensors_train[i_train][0] = prediction
        patient_vs_weight_tensors_train[i_train][0] = sample_weight
        i_train = i_train + 1

        ######### rotate along HxW plane ###############
        angle = np.random.randint(low=-15, high=+15)
        image_array_adc_temp = rotate(image_array_adc, angle=angle, axes=(2, 3), reshape=False)
        image_array_hbv_temp = rotate(image_array_hbv, angle=angle, axes=(2, 3), reshape=False)
        image_array_t2w_temp = rotate(image_array_t2w, angle=angle, axes=(2, 3), reshape=False)

        patient_vs_adc_tensors_train[i_train][0, 0:image_array_adc.shape[1],0:image_array_adc.shape[2],0:image_array_adc.shape[3]] = image_array_adc_temp
        patient_vs_hbv_tensors_train[i_train][0, 0:image_array_hbv.shape[1],0:image_array_hbv.shape[2],0:image_array_hbv.shape[3]] = image_array_hbv_temp
        patient_vs_t2w_tensors_train[i_train][0, 0:image_array_t2w.shape[1],0:image_array_t2w.shape[2],0:image_array_t2w.shape[3]] = image_array_t2w_temp
        patient_vs_prediction_tensors_train[i_train][0] = prediction
        patient_vs_weight_tensors_train[i_train][0] = sample_weight
        i_train = i_train + 1

        # gaussian Noise ################
        image_array_adc_temp= image_array_adc + np.random.normal(0, 0.01, image_array_adc.shape)
        image_array_hbv_temp = image_array_hbv + np.random.normal(0, 0.01, image_array_hbv.shape)
        image_array_t2w_temp = image_array_t2w + np.random.normal(0, 0.01, image_array_t2w.shape)

        patient_vs_adc_tensors_train[i_train][0, 0:image_array_adc.shape[1],0:image_array_adc.shape[2],0:image_array_adc.shape[3]] = scale_image(image_array_adc_temp)
        patient_vs_hbv_tensors_train[i_train][0, 0:image_array_hbv.shape[1],0:image_array_hbv.shape[2],0:image_array_hbv.shape[3]] = scale_image(image_array_hbv_temp)
        patient_vs_t2w_tensors_train[i_train][0, 0:image_array_t2w.shape[1],0:image_array_t2w.shape[2],0:image_array_t2w.shape[3]] = scale_image(image_array_t2w_temp)
        patient_vs_prediction_tensors_train[i_train][0] = prediction
        patient_vs_weight_tensors_train[i_train][0] = sample_weight
        i_train = i_train + 1


    # now convert to tensor

    patient_vs_adc_tensors_train = torch.tensor(patient_vs_adc_tensors_train, dtype=torch.float)
    patient_vs_hbv_tensors_train = torch.tensor(patient_vs_hbv_tensors_train, dtype=torch.float)
    patient_vs_t2w_tensors_train = torch.tensor(patient_vs_t2w_tensors_train, dtype=torch.float)
    patient_vs_prediction_tensors_train = torch.tensor(patient_vs_prediction_tensors_train, dtype=torch.float)
    patient_vs_weight_tensors_train = torch.tensor(patient_vs_weight_tensors_train, dtype=torch.float)


    return patient_vs_adc_tensors_train, patient_vs_hbv_tensors_train, patient_vs_t2w_tensors_train, patient_vs_prediction_tensors_train, patient_vs_weight_tensors_train


def data_to_3Dtensor(
    metadata_adc,
    metadata_hbv,
    metadata_t2w,
    patient_vs_modality_vs_image, 
    patient_vs_timeBCR,
    patient_vs_weight,
    test_fold_patient_list,
    validation_fold_patient_list,
    validation_percent = -1
    ):
    """
    This model takes the training data in raw format and convert them to tensors.

    Args:
        metadata_adc (list): List of channel, height, width information for ADC slides,
        metadata_hbv (list): List of channel, height, width information for HBV slides,
        metadata_t2w (list): List of channel, height, width information for T2W slides,
        patient_vs_modality_vs_image (dictionary): It has image array per each modality for each patient, 
        patient_vs_timeBCR (dictionary): It has time to BCR information for each patient,
        test_fold_patient_list (list): It is a list for testing patient for current fold,
        validation_percent (int): Percentage of training data (four folds) to keep for validation. 

    Return:
        tensor format of training_set, validation_set, testing_set. It also returns the patient_order in 
        testing set which is used for logging case by case report for each fold.

    """


    ####### initialze placeholder arrays that will be converter to tensor #######

    testing_data_count = len(test_fold_patient_list)
    validation_data_count = len(validation_fold_patient_list)
    if len(validation_fold_patient_list)!=0:
        training_data_count = len(list(patient_vs_modality_vs_image.keys())) - testing_data_count - validation_data_count
    else:
        training_data_count = len(list(patient_vs_modality_vs_image.keys())) - testing_data_count


    patient_vs_adc_tensors_train = np.zeros((training_data_count, 1, metadata_adc[0], metadata_adc[1], metadata_adc[2]))
    patient_vs_adc_tensors_test = np.zeros((testing_data_count, 1, metadata_adc[0], metadata_adc[1], metadata_adc[2]))
    patient_vs_adc_tensors_validation = np.zeros((validation_data_count, 1, metadata_adc[0], metadata_adc[1], metadata_adc[2]))


    patient_vs_hbv_tensors_train = np.zeros((training_data_count, 1, metadata_hbv[0], metadata_hbv[1], metadata_hbv[2]))
    patient_vs_hbv_tensors_test = np.zeros((testing_data_count, 1, metadata_hbv[0], metadata_hbv[1], metadata_hbv[2]))
    patient_vs_hbv_tensors_validation = np.zeros((validation_data_count, 1, metadata_hbv[0], metadata_hbv[1], metadata_hbv[2]))

    patient_vs_t2w_tensors_train = np.zeros((training_data_count, 1, metadata_t2w[0], metadata_t2w[1], metadata_t2w[2]))
    patient_vs_t2w_tensors_test = np.zeros((testing_data_count, 1, metadata_t2w[0], metadata_t2w[1], metadata_t2w[2]))
    patient_vs_t2w_tensors_validation = np.zeros((validation_data_count, 1, metadata_t2w[0], metadata_t2w[1], metadata_t2w[2]))

    patient_vs_prediction_tensors_train = np.zeros((training_data_count, 1))
    patient_vs_prediction_tensors_test = np.zeros((testing_data_count,1))
    patient_vs_prediction_tensors_validation = np.zeros((validation_data_count,1))

    patient_vs_weight_tensors_train = np.zeros((training_data_count, 1))

    i_test = 0
    i_train = 0
    i_val = 0
    patient_order = []
    for patient_id in patient_vs_modality_vs_image:

        image_array_adc = patient_vs_modality_vs_image[patient_id]['adc']
        image_array_hbv = patient_vs_modality_vs_image[patient_id]['hbv']
        image_array_t2w = patient_vs_modality_vs_image[patient_id]['t2w']

        prediction = patient_vs_timeBCR[patient_id]
        sample_weight = patient_vs_weight[patient_id]

        if int(patient_id) in test_fold_patient_list:
            patient_vs_adc_tensors_test[i_test][0, 0:image_array_adc.shape[0],0:image_array_adc.shape[1],0:image_array_adc.shape[2]] = scale_image(image_array_adc)
            patient_vs_hbv_tensors_test[i_test][0, 0:image_array_hbv.shape[0],0:image_array_hbv.shape[1],0:image_array_hbv.shape[2]] = scale_image(image_array_hbv)
            patient_vs_t2w_tensors_test[i_test][0, 0:image_array_t2w.shape[0],0:image_array_t2w.shape[1],0:image_array_t2w.shape[2]] = scale_image(image_array_t2w)
            patient_vs_prediction_tensors_test[i_test][0] = prediction
            i_test = i_test + 1
            patient_order.append(patient_id)

        elif int(patient_id) in validation_fold_patient_list:
            patient_vs_adc_tensors_validation[i_val][0, 0:image_array_adc.shape[0],0:image_array_adc.shape[1],0:image_array_adc.shape[2]] = scale_image(image_array_adc)
            patient_vs_hbv_tensors_validation[i_val][0, 0:image_array_hbv.shape[0],0:image_array_hbv.shape[1],0:image_array_hbv.shape[2]] = scale_image(image_array_hbv)
            patient_vs_t2w_tensors_validation[i_val][0, 0:image_array_t2w.shape[0],0:image_array_t2w.shape[1],0:image_array_t2w.shape[2]] = scale_image(image_array_t2w)
            patient_vs_prediction_tensors_validation[i_val][0] = prediction
            i_val = i_val + 1

        else:
            patient_vs_adc_tensors_train[i_train][0, 0:image_array_adc.shape[0],0:image_array_adc.shape[1],0:image_array_adc.shape[2]] = scale_image(image_array_adc)
            patient_vs_hbv_tensors_train[i_train][0, 0:image_array_hbv.shape[0],0:image_array_hbv.shape[1],0:image_array_hbv.shape[2]] = scale_image(image_array_hbv)
            patient_vs_t2w_tensors_train[i_train][0, 0:image_array_t2w.shape[0],0:image_array_t2w.shape[1],0:image_array_t2w.shape[2]] = scale_image(image_array_t2w)
            patient_vs_prediction_tensors_train[i_train][0] = prediction
            patient_vs_weight_tensors_train[i_train][0] = sample_weight
            i_train = i_train + 1

    # take 20% of train for validation ########################################################
    
    if validation_percent != -1:
        sample_count = training_data_count
        row_perm = torch.randperm(sample_count)
        patient_vs_adc_tensors_train = patient_vs_adc_tensors_train[row_perm]
        patient_vs_hbv_tensors_train = patient_vs_hbv_tensors_train[row_perm]
        patient_vs_t2w_tensors_train = patient_vs_t2w_tensors_train[row_perm]
        patient_vs_prediction_tensors_train = patient_vs_prediction_tensors_train[row_perm]
        patient_vs_weight_tensors_train = patient_vs_weight_tensors_train[row_perm]
        
        validation_count = (training_data_count*validation_percent) // 100
        patient_vs_adc_tensors_validation = patient_vs_adc_tensors_train[0:validation_count]
        patient_vs_adc_tensors_train = patient_vs_adc_tensors_train[validation_count:]
        
        patient_vs_hbv_tensors_validation = patient_vs_hbv_tensors_train[0:validation_count]
        patient_vs_hbv_tensors_train = patient_vs_hbv_tensors_train[validation_count:]

        patient_vs_t2w_tensors_validation = patient_vs_t2w_tensors_train[0:validation_count]
        patient_vs_t2w_tensors_train = patient_vs_t2w_tensors_train[validation_count:]

        patient_vs_prediction_tensors_validation = patient_vs_prediction_tensors_train[0:validation_count]
        patient_vs_prediction_tensors_train = patient_vs_prediction_tensors_train[validation_count:]

        #patient_vs_weight_tensors_validation = patient_vs_weight_tensors_train[0:validation_count]
        patient_vs_weight_tensors_train = patient_vs_weight_tensors_train[validation_count:]        

    ############################################################################################
    # convert to tensor
    #patient_vs_adc_tensors_train = torch.tensor(patient_vs_adc_tensors_train, dtype=torch.float)
    patient_vs_adc_tensors_validation = torch.tensor(patient_vs_adc_tensors_validation, dtype=torch.float)
    patient_vs_adc_tensors_test = torch.tensor(patient_vs_adc_tensors_test, dtype=torch.float)

    #patient_vs_hbv_tensors_train = torch.tensor(patient_vs_hbv_tensors_train, dtype=torch.float)
    patient_vs_hbv_tensors_validation = torch.tensor(patient_vs_hbv_tensors_validation, dtype=torch.float)
    patient_vs_hbv_tensors_test = torch.tensor(patient_vs_hbv_tensors_test, dtype=torch.float)

    #patient_vs_t2w_tensors_train = torch.tensor(patient_vs_t2w_tensors_train, dtype=torch.float)
    patient_vs_t2w_tensors_validation = torch.tensor(patient_vs_t2w_tensors_validation, dtype=torch.float)
    patient_vs_t2w_tensors_test = torch.tensor(patient_vs_t2w_tensors_test, dtype=torch.float)

    #patient_vs_prediction_tensors_train = torch.tensor(patient_vs_prediction_tensors_train, dtype=torch.float)
    patient_vs_prediction_tensors_validation = torch.tensor(patient_vs_prediction_tensors_validation, dtype=torch.float)
    patient_vs_prediction_tensors_test = torch.tensor(patient_vs_prediction_tensors_test, dtype=torch.float)


    #patient_vs_weight_tensors_train = torch.tensor(patient_vs_weight_tensors_train, dtype=torch.float)
    #patient_vs_weight_tensors_validation = torch.tensor(patient_vs_weight_tensors_validation, dtype=torch.float)


    training_set = [patient_vs_adc_tensors_train, patient_vs_hbv_tensors_train, patient_vs_t2w_tensors_train, patient_vs_prediction_tensors_train, patient_vs_weight_tensors_train]
    testing_set = [patient_vs_adc_tensors_test, patient_vs_hbv_tensors_test, patient_vs_t2w_tensors_test, patient_vs_prediction_tensors_test]
    validation_set = [patient_vs_adc_tensors_validation, patient_vs_hbv_tensors_validation, patient_vs_t2w_tensors_validation, patient_vs_prediction_tensors_validation]
    ############################################################################################

    return training_set, validation_set, testing_set, patient_order
