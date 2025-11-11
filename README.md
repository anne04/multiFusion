This repository develops a multimodal deep learning model 'multiFusion' for predicting time to biochemical recurrence by integrating MRI scans coming from different modalities, in particular, Apparent Diffusion Coefficient (ADC) maps, High b-value Diffusion (HBV),  and T2-weighted imaging with prostate mask. 

## Packages to install:
Please see [this](https://github.com/anne04/multiFusion/blob/main/package_installation.md) to setup the environment by installing required Python packages.


## Running MultiFusion model
Please pull this repository, download the dataset "Multimodal-Quiz", keep it inside the repository directory, and set it as current working directory. Then follow the steps as below:

### Preprocessing Step:
This step takes the image arrays from three modalities and time to BCR from clinical metadata and combine them into one pickle file for the training purpose. 

```
nohup python -u data_preprocess.py --path_dataset_mpMRI=Multimodal-Quiz/radiology/mpMRI/ --path_dataset_prost_mask_t2w=Multimodal-Quiz/radiology/prostate_mask_t2w/ --path_dataset_clinical=Multimodal-Quiz/clinical_data/ --output_path=Multimodal-Quiz/ > output_preprocess.log &
```

This command will preprocess the input dataset and produce following two files:
1. Multimodal-Quiz/training_data_multiFusion.pkl (Dataset for model training)
2. Multimodal-Quiz/patient_vs_clinical_data.csv (For easy visualization of metadata)

### Training Step and K-fold cross-validation:
```
nohup python -u run_multiFusion.py --training_data=Multimodal-Quiz/training_data_multiFusion.pkl --model_name=2DmultiFusion_test --conv_dimension=2 --wandb_project_name=2Dmultimodal_fusion' --fold_count=5 --kfold_info_file=Multimodal-Quiz/data_split_5fold.csv 
```

This command will run the multimodal fusion model that takes MRI scans (image array) from three different modalities: ADC & HBV scans, and T2W prostate (masked by additional mask), extracts features from each modality separately (Convolution Neural Network), fuses them together (Multi-Layer Perceptron) to get an integrated feature embedding, and finally predicts the time to BCR (Multi-Layer Perceptron). The parameter --conv_dimension=2 performs 2D convolution, but user can use --conv_dimension=3 for running a 3D convolution for feature extraction. Other available parameters can be found in the User Guide. This step additionally runs K=5 fold cross-validation, and produces the performance, i.e., the C-index for each fold and case-by-case study reports. This step outputs the following:

1. Logs (loss curves) will be saved under 'wandb/' (offline mode) and 'output/' directories by default. You can sync the wandb outputs to your wandb account later for visualization of loss curves.
2. Best snapshot of the trained model will be saved under 'model/' directory.
3. The C-index for each fold and case-by-case study reports are generated in 'output/<model_name>_fold_vs_c_index_report.csv' and 'output/<model_name>_case_by_case_report.csv' respectively. 

#### Evaluation Results:
A sample wandb log for the above run can be viewed [here](https://github.com/anne04/multiFusion/blob/main/wandb_log_multiFusion_2dConv_model.png). The corresponsing C-index report for each fold is [here](https://github.com/anne04/multiFusion/blob/main/fold_vs_c_index_report.csv) and the case-by-case study for the patients is [here](https://github.com/anne04/multiFusion/blob/main/case_by_case_report.csv). (Please note that, the goal of this project is to demonstrate the development of a multimodal model, thus the used dataset is very small (95 patients). That is why the accuracy is not very impressive. But in a real case scenario, the dataset can be larger, and pretrained foundation models will be used for feature extraction instead of developing a feature extractor from scratch, which is supposed to improve the results drastically.  

### User Parameters:
You can run the python scripts with -h parameter to see all available parameters. These are also provided [here](https://github.com/anne04/multiFusion/blob/main/user_parameters.md)


