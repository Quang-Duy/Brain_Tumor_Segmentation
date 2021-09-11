
# Brain Tumor Segmentation

Brain Tumor Segmentation provides characterization & measurement of glioma sub-regions to aid doctors with treatment decisions. ResNet50 model is used for this segmentation task with MLFlow to track experiments record.

Directory Structure
------------

The directory structure of the Brain Tumor Segmentation project looks like this:
```
├── README.md                               <- The top-level README for developers using this project.
├── bmri                                    <- All scripts to run this project
│   ├── data                                <- Scripts to generate data
|   |   ├── dataset.py
|   |   └── make_dataset.py
│   ├── transform                           <- Script to augment data
|   |   └── image_transform.py
│   └── model                               <- Scripts to build, train and evals models
|       ├── skull_stripping                 <- Scripts to remove skull from image
|       ├── bmri_eval.py
|       ├── bmri_metrics.py
|       ├── bmri_train.py
|       ├── bmri_UNet_model.py
|       ├── bmri_UNet_resnet50_model.py
|       ├── bmri_Unet_Vgg19_model.py
|       └── make_model.py
├── models                                  <- Location where all models are saved under
├── notebooks                               <- Quick look at dataset analysis
├── requirements.txt                        <- Requirements to download before executing project
└── params.yaml                             <- Contains parameters of model to load and train with