# Evaluation
If you want to replicate the results from our preprint check out notebooks for the [AMOS](Evaluate_on_AMOS22.ipynb) and the [TotalSegmentator MRI](Evaluate_on_TotalSegmentator-MRI.ipynb) datasets. Alternatively, if you have your own annotated dataset you can use [eval_data.py](eval_data.py).

### Ensemble VS single fold performance
|         | Ensemble Prediction | Single Fold Prediction |
| :-------- | :-------: | :-------: | 
|Avg. DSC (NAKO GRE MRI) | 0.88  | 0.87 | 
|Avg. DSC (NAKO T2-HASTE MRI) | 0.85  | 0.83| 
|Avg. time per scan*| 33s  | 23s| 

*We used an Nvidia A100 to segment NAKO GRE whole body scans

Most of the computation time is needed for resampling the images on the CPU, not by the segmentation itself. 
Single fold prediction is therefore only around 30% faster on our system.


### TotalSegmentator supports MRI now as well, how does it compare?
We added a comparisson of both models to our [preprint](https://arxiv.org/pdf/2405.06463).
In the NAKO dataset, MRSegmentator consistently outperforms TotalSegmentator MRI across all classes and sequences. 
This could partly be due to annotation bias since the same radiologists worked on our training and test data. 
Regardless, MRSegmentator achieves superior DSC values for each class in the AMOS dataset, too.
Even on TotalSegmentator’s MRI dataset, which may have an annotation bias favoring their model, MRSegmentator shows a comparable average DSC (0.74 vs. 0.76) and offers improved predictions for 18 classes, especially for abdominal organs and blood vessels.

That said, TotalSegmentator MRI supports a broader array of structures, particularly muscle groups in the femoral region, such as the quadriceps femoris, thigh compartments, and sartorius muscles. If your research focuses on these structures check out their new model!

### MRSegmentator v1.2: What changed?
We moved the NAKO MRI from the test partition to the training partition and retrained the model.
This resulted in better segmentation performance for NAKO data, while the performance on the other datasets remained the same. Read more in our [notebook](Compare_Versions.ipynb).

|Dataset|MRSegmentator v1.1|MRSegmentator v1.2|
| :-------- | :-------: | :-------: |
| NAKO GRE | 0.88 | 0.91 |
| NAKO T2-HASTE | 0.85 | 0.89 |
| AMOS MRI | 0.79 | 0.79 |
| AMOS CT | 0.84 | 0.84 |
| TotalSegmentator MRI data  |0.74 | 0.74 |