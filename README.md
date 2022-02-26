# Visual Transformers for the Segmentation of Liver and Liver Tumor in CT Scans of Human Abdomen

### Description
tbd

### Data Folder Structure
The structure is as follows: We want to store the volumes used in training (training and validation sets)
separately from the testing ones.

There is a method (`pre_process_niis`) that extracts slices and masks from .nii files and store them in corresponding (`data/train-val/vols-2d/` and `data/train-val/segs-2d/` folders).
These 2D images are then used directly in Datasets and Data Loaders to train the networks.

```
.
│   
└─── src
│      │ .
│      │ .
│      │ .  
│
└─── data
       │ 
       └─── train-val
       │       │
       │       └───  vols-2d
       │       │       │ volume-1-0.png
       │       │       │ volume-1-1.png
       │       │       │ .
       │       │       │ .
       │       │
       │       └─── vols-3d
       │       │       │ volume-1.nii
       │       │       │ volume-2.nii
       │       │       │ .
       │       │       │ .
       │       │
       |       |
       |       |
       │       └───  segs-2d
       │       │       │ segmentation-1-0.png
       │       │       │ segmentation-1-1.png
       │       │       │ .
       │       │       │ .
       │       │
       │       └─── segs-3d
       │               │ segmentation-1.nii
       │               │ segmentation-2.nii
       │               │ .
       │               │ .
       │   
       └─── test
               │ tbd
```