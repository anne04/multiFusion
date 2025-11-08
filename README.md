This repository presents a deep learning model for integrating MRI scans coming from different modalities, in particular, ADC, HBV, and T2W. 


Instructions for running the model are as follows:
(Assuming you have pulled this repository and set it as current working directory)
## Preprocessing Step:
```
nohup python -u data_preprocess.py --path_dataset_mpMRI=Multimodal-Quiz/radiology/mpMRI/ --path_dataset_prost_mask_t2w=Multimodal-Quiz/radiology/prostate_mask_t2w/
```
