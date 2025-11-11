### data_preprocess.py 
```
--path_dataset_mpMRI = Path to mpMRI data, type=str, default='Multimodal-Quiz/radiology/mpMRI/'

--path_dataset_prost_mask_t2w = Path to the mask image for T2W prostate scan, type=str, default="Multimodal-Quiz/radiology/prostate_mask_t2w/"

--path_dataset_clinical = Path to the clinical metadata files, type=str, default='Multimodal-Quiz/clinical_data/'

--output_path = Path to the output directory, type=str, default='Multimodal-Quiz/'
```
### run_multiFusion.py
```
--training_data = 'Path to training data', type=str, default='Multimodal-Quiz/training_data_multiFusion.pkl'

--model_name = Provide a model name, type=str, default="2DmultiFusion_test"

--wandb_project_name = Provide a project name for wandb log generation, type=str, default='2Dmultimodal_fusion'

--fold_count = Provide the total fold, type=int, default=5

--kfold_info_file = Provide the file name with k fold info, type=str, default="Multimodal-Quiz/data_split_5fold.csv"

--conv_dimension = Set to 2 or 3, for running 2D or 3D convolution operation respectively for feature extraction, type=int, default=2

--num_epoch = Number of epochs or iterations for model training, type=int, default=1000

--model_path = Path to save the model state, type=str, default='model/'

--output_path = Path to save the final performance reports and loss curves in csv format, type=str, default='output/'

--dropout = Set a dropout value, type=float, default=0.5

--batch_size = Set the minibatch size for model training, type=int, default=10

--lr_rate = Set the learning rate for model training, type=float, default=0.001

--manual_seed = Set it to yes for reproducible result, type=str, default='no'

--seed = Set it to some integer for reproducible result, type=int.
```

