# Automatic segmentation of abdominal MRI scans
> Detect and segment 40 classes in MRI scans of the abdominal / pelvic / chest region

Contrary to CT scans, where tools for automatic multi-structure segmentation are quite mature ([TotalSegmentator](https://github.com/wasserth/TotalSegmentator) / [Totalsegmentator Mini](https://github.com/kbressem/totalsegmentator-mini)), segmentation tasks in MRI scans are often either focused on the brain region or on a subset of few organs in other body regions. This project aims to extend this and accurately segment 40 organs and structures in human MRI scans of the abdominal, pelvic and chest regions. The segmenation works well on different sequence types, including T1- and T2-weighted, fat-saturated and Dixon sequences.

## Datawrangling and Preprocessing
All steps on datawrangling and preprocessing are documented in the notebook section. We use data from two dataset:
(1) T2-weighted Dixon sequences from the UK Biobank (restricted public availability) 
(2) T1, T2 and T1fs-postKM sequences of the Charié kidney dataset (not publicly available) 

A radiology resident in our team annotated X scans in our dataset. For this she used an AI-assisted interactive approach within the MonaiLabel framework. That is, we incrementally trained a U-Net to predict segmentation masks that helped the radiologis to increase her annotation speed.
Additionally, we used TotalSegmentator to create initial segmentation masks for the UKBB data. TotalSegmentator is intended for CT scans and does not reliably segment most MRI scans, however, we found that it works suprisingly well on MRI sequences similar to CT data, such as Dixon water-only sequences. We then made use of the special property of Dixon sequences (i.e. different sequences share an (almost) identical latent space) and registered the segmentation masks from water-only sequences to other sequence types (i.e. fat-only, in-phase, opposing-phase). 

First we trained a U-Net on this set of initial UKBB segmentations,  using a self-supverised strategy. Then, we finetuned the model on our set of manual annotation. Using the new model we created updated segementation mask and repeated this process of self-supervised learning / finetuning a few times. 



## Overview of bundle commands
### Inference

```bash
python -m monai.bundle run inference \
  --meta_file configs/metadata.json \
  --config_file configs/inference.yaml \
  --logging_file configs/logging.conf \
  --dataset_dir <path_to_file_or_folder>
```
To run inference on a single file or a directory containing multiple image files use the `--dataset_dir` flag. 
Nested directories are supported up to a depth of 1.   
To control where the files are stored, overwrite the output directory with the `--output_dir` flag. 

### Evaluation
```bash
python -m monai.bundle run evaluating \
  --meta_file configs/metadata.json \
  --config_file configs/evaluate.yaml \
  --logging_file configs/logging.conf
```

### Training

During training, this bundle saves both, the model weights AND the optimizer in `model.pt`. This can be an issue, e.g. if deployed in MONAI Label. Use `scripts/separate_model_optim.py` to separate them. 

#### Single GPU training

```bash
python -m monai.bundle run training \
  --meta_file configs/metadata.json \
  --config_file "['configs/train.yaml','configs/unet.yaml']" \
  --logging_file configs/logging.conf
```


## Class details

|Index|Class|
| :-------- | :------- |
| 0 | background |
| 1 | spleen |
| 2 | right_kidney |
| 3 | left_kidney |
| 4 | gallbladder |
| 5 | liver |
| 6 | stomach |
| 7 | pancreas |
| 8 | right_adrenal_gland |
| 9 | left_adrenal_gland |
| 10 | left_lung |
| 11 | right_lung |
| 12 | heart |
| 13 | aorta |
| 14 | inferior_vena_cava |
| 15 | portal_vein_and_splenic_vein |
| 16 | left_iliac_artery |
| 17 | right_iliac_artery |
| 18 | left_iliac_vena |
| 19 | right_iliac_vena |
| 20 | esophagus |
| 21 | small_bowel |
| 22 | duodenum |
| 23 | colon |
| 24 | urinary_bladder |
| 25 | spine |
| 26 | sacrum |
| 27 | left_hip |
| 28 | right_hip |
| 29 | left_femur |
| 30 | right_femur |
| 31 | left_autochthonous_muscle |
| 32 | right_autochthonous_muscle |
| 33 | left_iliopsoas_muscle |
| 34 | right_iliopsoas_muscle |
| 35 | left_gluteus_maximus |
| 36 | right_gluteus_maximus |
| 37 | left_gluteus_medius |
| 38 | right_gluteus_medius |
| 39 | left_gluteus_minimus |
| 40 | right_gluteus_minimus |




## Workplan

### Annotation
- Add remaining 20 classes to annotations of Charité dataset
- Increase number of annotated UKBB scans
- Annoate section 0 (upper chest + shoulders) and 4 (thighs)
   - Section 0 includes part of the lungs and spine
   - Section 4 includes part of the femurs

### Programming
- Include Charité images to training pipeline
- Create a docker image
- Create a baseline based on the nnU-Net framework

### Experimental Setting
- Find and annotate an external dataset
- Results must be evaluated by a second radiologist 
   - Visual evaluation
   - Dice of annotation-overlap between two radiologists (if we have enough time)