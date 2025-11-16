This repository develops a multimodal deep learning model 'multiFusion' for predicting time to biochemical recurrence by integrating MRI scans coming from different modalities, in particular, Apparent Diffusion Coefficient (ADC) maps, High b-value Diffusion (HBV),  and T2-weighted imaging with prostate mask. 

## Packages to install:
This project is developed on Ubuntu environment with four CPUs each having 30GB RAM, 1 GPU with 12 GB GPU memory, and Python 3.10.13, Pytorch 2.2.2 (with CUDA 12.1). For more details to setup the environment 
by installing required Python packages please see [this](https://github.com/anne04/multiFusion/blob/main/package_installation.md) 

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

This command will run the multimodal fusion model that takes input MRI scans (image array) from three different modalities: ADC & HBV scans, and T2W prostate (masked by additional mask). It calls training utility functions to scale each modality image  to 0-255 range, and handling the missing slides or dimension mismatch by zero-padding. Then the multiFusion model is initialized that extracts features from each modality separately (Convolution Neural Network), fuses them together (Multi-Layer Perceptron) to get an integrated feature embedding, and finally predicts the time to BCR (Multi-Layer Perceptron). The parameter --conv_dimension=2 performs 2D convolution, but user can use --conv_dimension=3 for running a 3D convolution for feature extraction. Other available parameters can be found [here](https://github.com/anne04/multiFusion/blob/main/user_parameters.md). This step additionally runs K=5 fold cross-validation, and produces the performance, i.e., the C-index for each fold and case-by-case study reports. This step outputs the following:

1. Logs (loss curves) will be saved under 'wandb/' (offline mode) and 'output/' directories by default. You can sync the wandb outputs to your wandb account later for visualization of loss curves.
2. Best snapshot of the trained model will be saved under 'model/' directory.
3. The C-index for each fold and case-by-case study reports are generated in 'output/<model_name>_fold_vs_c_index_report.csv' and 'output/<model_name>_case_by_case_report.csv' respectively. 

### Inference and report generation:
If you quickly want to run the inference on the data (download [here](https://huggingface.co/fatema04/multiFusion/tree/main)) and generate fold-wise reports then run the following command:
```
nohup python -u inference_and_report.py --training_data=Multimodal-Quiz/training_data_multiFusion.pkl --model_name=2DmultiFusion_test --model_path=submission_package_models/ --conv_dimension=2 --wandb_project_name=2Dmultimodal_fusion' --fold_count=5 --kfold_info_file=Multimodal-Quiz/data_split_5fold.csv --output=submission_package_models/output/ 
```
It should print the following output:
 ![](https://github.com/anne04/multiFusion/blob/main/inference.png)


### Trained Models:
Trained models can be found [here](https://huggingface.co/fatema04/multiFusion/tree/main).

### Evaluation Results:
A sample wandb log for the above run can be viewed [here](https://github.com/anne04/multiFusion/blob/main/wandb_log_multiFusion_2dConv_model.png). The corresponding C-index report for each fold is [here](https://github.com/anne04/multiFusion/blob/main/2DmultiFusion_test_fold_vs_c_index_report.csv) and the case-by-case study for the patients is [here](https://github.com/anne04/multiFusion/blob/main/2DmultiFusion_test_case_by_case_report.csv). (Please note that, in a real case scenario, pretrained foundation models will be used for feature extraction instead of developing a feature extractor from scratch, which is supposed to improve the results.) 

### User Parameters:
You can run the python scripts with -h parameter to see all available parameters. These are also provided [here](https://github.com/anne04/multiFusion/blob/main/user_parameters.md)


