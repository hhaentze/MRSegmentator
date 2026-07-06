# Changelog

<!--next-version-placeholder-->


## v2.0.0 (06/07/2027)

### Feature
- 10 additional classes for body composition in MRI
- weights management supports muliple weights
- dedicatated weights directory as ~/.mrsegmentator
- env variable MRSEG_WEIGHTS_PATH allows to link to a custom weights dir
- if MRSEG_WEIGHTS_PATH points to any nnUNet results direcotry, these weights can be loaded as well
- More CI tests, with full integration workflows
- More GitHub action workflows
- Updated supported python versions to 3.10-3-13. (3.9 is not working)
- Added fast inference mode with `--fast`


## v1.3.2 (26/05/2026)

### Fix
- Fixed incorrect handling of image orientations such as PIL or ASL

## v1.3.1 (11/08/2025)

### Fix
- fixed incorrect file names that occured when mapping DICOM to NIfTI

## v1.3.0 (09/08/2025)

### Feature
- Automatically run on CPU if no GPU was detected
- Add support for .mha and .nrrd
- Add support for DICOM and DICOM SEG
- Enable better logging:
    - control log lebel (DEBUG, INFO, WARNING)
    - add flag --no_tqdm to disable progress bars
    - remove --verbose flag (replaced by --log_level DEBUG)

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