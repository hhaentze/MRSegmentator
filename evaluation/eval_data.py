# Copyright 2024 Hartmut Häntze

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from os.path import join

import pandas as pd
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    MapLabelValued,
    Spacingd,
)
from tqdm import tqdm

import mappings

# preprocessing
ConvolutedLoad = lambda class_mapping, spacing: Compose(
    [
        LoadImaged(keys=["pred", "label"], image_only=True),
        EnsureChannelFirstd(keys=["pred", "label"]),
        Spacingd(
            keys=["pred", "label"],
            pixdim=spacing,
            mode="nearest",
        ),
        MapLabelValued(
            keys=["label"],
            orig_labels=class_mapping.keys(),
            target_labels=class_mapping.values(),
        ),
        AsDiscreted(keys=["pred", "label"], to_onehot=41),
        Lambdad(keys=["pred", "label"], func=lambda x: x[None]),
    ]
)


def calc_metric(data, class_mapping, spacing):
    """Calculate DSC and Hausdorff distance. Returns tuple with both metrics"""

    # define metric
    dice_metric = DiceMetric(include_background=False, reduction="none", num_classes=41)
    haus_metric = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)

    # define data and preprocessing
    data_list = data[["label", "pred"]].to_dict("records")
    load = ConvolutedLoad(class_mapping, spacing)

    # iterate over all items
    for item in tqdm(data_list, desc='Calc. Dice"'):
        _item = load(item)
        dice_metric(_item["pred"], _item["label"])
        with warnings.catch_warnings(action="ignore"):
            haus_metric(_item["pred"], _item["label"], spacing=spacing)

    # transform results to dataframe
    scores_dice = pd.DataFrame(dice_metric.get_buffer())
    scores_haus = pd.DataFrame(haus_metric.get_buffer())
    scores_dice.columns = mappings.channels
    scores_haus.columns = mappings.channels

    return scores_dice, scores_haus


def main(
    label_dir: str,
    seg_dir: str,
    dataset_name: str,
    class_mapping=mappings.amos_to_mrseg,
    spacing=(1.5, 1.5, 3.0),
):
    """evaluate prediction with regard to groundtruth, save dsc and hausdorff distance to csv file.

    label_dir: path to groundtruth directory
    seg_dir: path to segmentation directory. Files need to have same name as groundtruths (with or without '_seg' postfix).
    dataset_name: Results will be stored in files f"{dataset_name}_dsc.csv" and f"{dataset_name}_hd.csv"
    class_mapping: dictionary that maps classes to the 40 labels of MRSegmentator. See mappings.py
    spacing: spacing for DSC calculation. Increase spacing to reduce computation time
    """

    # Read files
    names = [f.name for f in os.scandir(seg_dir) if f.name[-7:] == ".nii.gz"]
    names.sort()

    # Save file names in data frame
    data = pd.DataFrame()
    data["pred"] = [join(seg_dir, n) for n in names]
    data["label"] = [join(label_dir, n.replace("_seg", "")) for n in names]

    # Calculate Dice and Hausdorff distance
    scores_dice, scores_haus = calc_metric(data, class_mapping, spacing)

    # Save results
    scores_dice.to_csv(dataset_name + "_dsc.csv", index=False)
    scores_haus.to_csv(dataset_name + "_hd.csv", index=False)
