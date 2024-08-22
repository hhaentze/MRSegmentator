# Add NAKO to the training pipeline
In our [preprint](https://arxiv.org/pdf/2405.06463) (version 1.1) we evaluate MRSegmentator on external data, which we excluded from the training pipeline.
Afterwards, we retrained the model on both training and parts of the test data (i.e. NAKO dataset) to further increase performance (version 1.2). If you wish to use the model, trained without the external NAKO data, you can specify the version to 1.1 during installation with pip.

## Download evaluation datasets
- AMOS: https://zenodo.org/records/7262581
- TotalSegmentator-MRI: https://zenodo.org/records/11367005

MRI in the AMOS dataset start with from the id 500. We exlude urinary bladder and prostate/uterus.
The TotalSegmentator-MRI dataset includes annotation masks for each class. Before evaluation we merge all annotation masks into a single file for every image. For a fair comparison with TotalSegmentator-MRI we only evaluate on the data marked as test-data.

## Metrics
Calculate Dice with Monai. See [eval_data.py](eval_data.py).

## Results
|Dataset|MRSegmentator v1.1|MRSegmentator v1.2|
| :-------- | :-------: | :-------: |
| AMOS MRI | 0.79 | 0.79 |
| AMOS CT | 0.84 | 0.84 |
| TotalSegmentator-MRI |0.72 | 0.74 |

Including the NAKO data to the training pipeline did not change our model's performance on the AMOS data but increased segmentation quality within the TotalSegmentator-MRI dataset. The biggest improvement can likely be found for the contrast types of the the NAKO data set (i.e. Dixon and T2-HASTE), but it is challenging to quantify this, as the data lost it's status as external dataset. 