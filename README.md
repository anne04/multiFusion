This repository presents a deep learning model for predicting time to biochemical recurrence by integrating MRI scans coming from different modalities, in particular, ADC, HBV, and T2W. 


Instructions for running the model are as follows:

(Assuming you have pulled this repository and set it as current working directory)
## Preprocessing Step:
This step takes the image arrays from three modalities and time to BCR from clinical metadata and combine them into one pickle file for the training purpose. 

```
nohup python -u data_preprocess.py --path_dataset_mpMRI=Multimodal-Quiz/radiology/mpMRI/ --path_dataset_prost_mask_t2w=Multimodal-Quiz/radiology/prostate_mask_t2w/ --path_dataset_clinical=Multimodal-Quiz/clinical_data/ --output_path=Multimodal-Quiz/ > output_preprocess.log &
```

This command will preprocess the input dataset and produce following two files:
1. Multimodal-Quiz/training_data_multiFusion.pkl (dataset for model training)
2. Multimodal-Quiz/patient_vs_clinical_data.csv (For easy visualization of metadata)

## Training Step and K-fold cross-validation:
```
nohup python -u run_multiFusion.py --training_data=Multimodal-Quiz/training_data_multiFusion.pkl --model_name=3DmultiFusion_test --conv_dimension=3 --wandb_project_name=3Dmultimodal_fusion' --fold_count=5 --kfold_info_file=Multimodal-Quiz/data_split_5fold.csv 
```

This command will run the multimodal fusion model that takes 3D MRI scans (image array) from three different modalities: ADC & HBV scans, and T2W prostate (masked by additional mask), extracts features from each modality separately (Convolution Neural Network), fuse them together (MLP) to get an integrated feature embedding, and finally predicts the time to BCR (MLP). The parameter --conv_dimension=3 ensures 3D convolution and user can use --conv_dimension=2 for running a 2D convolution for feature extraction. Logs (loss curves) will be saved under 'wandb/' and 'output/' directories by default. Best snapshot of the trained model will be saved under 'model/' directory. Other available parameters can be found in the User Guide.

This step additionally runs K=5 fold cross-validation, and the performance, i.e., the C-index for each fold and case-by-case study reports are generated in 'output/<model_name>_fold_vs_c_index_report.csv' and 'output/<model_name>_case_by_case_report.csv' respectively. 
