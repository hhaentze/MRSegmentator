# Changelog

<!--next-version-placeholder-->

## v1.2.3 (05/02/2025)

### Feature
- Print image and subvolume size if splitting is used
- Add option --split_margin to allow to change overlap between splitted volumes

### Fix
- Set pytotch version to <= 2.3.1
- Set python version to < 3.13
- ==> Fixes toch.pickle error due to updated dependency- Supress torch.load future warning, introduced by nnunet
- Increase default split_margin from 2 to 3

## v1.2.2 (11/12/2024)

### Feature
- Print segmentation time after finishing

### Fix
- Supress torch.load future warning, introduced by nnunet
- Print version number of custom weight directories, if they are specified

## v1.2.0 (22/08/2024)

### Feature
- Add NAKO data to training pipeline
- Update weights

### Fix
- Make ensemble prediction default for Python API

___

## v1.1.2 (24/06/2024)

### Fix
- Change python_requires from 3.11 to 3.9
- Remove monai dependency


## v1.1.0 (18/05/2024)

### Feature
- Update model weights with weights trained by `nnUNetTrainerNoMirroring`

### Fix
- Remove postprocessing `remap_left_right(...)`. It is not needed anymore.

___
## v1.0.0 (10/05/2024)
- First release of MRSegmentator