<h2 align="center"> DICOM Helper </h2> 

***

<div align="center">
<a href="https://github.com/hhaentze/MRSegmentator/actions"><img alt="Continuous Integration" src="https://github.com/hhaentze/MRSegmentator/actions/workflows/ci.yml/badge.svg"></a>
<a href="https://github.com/hhaentze/MRSegmentator/blob/master/License.txt"><img alt="License: Apache" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>  
<a href="https://pypi.org/project/mrsegmentator/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mrsegmentator"></a>  
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</div>

> Helper tool to convert DICOM to NIfTI and NIfTI to DICOM(SEG)

## Installation
It is part of the MRSegmentator package:
```bash
pip install mrsegmentator
```

## Conversion
### DICOM -> NIfTI
```bash
dcm_helper d2n 
   -i, --input <str> [required] # DICOM directory
   -o, --output <str>  # Output directory
   -s, --series <str>  # Specifc series ID
```

### NIfti -> DICOM SEG
```bash
dcm_helper n2seg 
   -i, --nifti <str> [required] # Input NIfTI segmentation
   -t, --template <str> [required] # Template DICOM directory
   -o, --output <str> [required] # Output DICOM SEG file
```