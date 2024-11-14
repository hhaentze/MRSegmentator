# Copyright 2024 Hartmut Häntze

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.import os

""" Files in the TotalSegmentator MRI dataset are stores one class per file. This script merges all classes into a single file """

import os
import sys
from os.path import join

import pandas as pd
from monai.transforms import LoadImage, SaveImage
from tqdm import tqdm

load = LoadImage()
Save = lambda outdir: SaveImage(
    output_dir=outdir,
    separate_folder=False,
    output_postfix="merged",
    print_log=False,
)


def main(path: str):

    assert os.path.isdir(path), "Requires path to TotalSegmentator-MRI dataset as input"

    data = pd.read_csv(join(path, "meta.csv"), sep=";")
    data = data.loc[data["split"] == "test"].reset_index(drop=True)
    total_channels = pd.read_csv("total_mri_classes.csv")

    for eid in tqdm(data["image_id"]):

        # Load first annotation
        label = load(join(path, eid, "segmentations", total_channels["class"][1] + ".nii.gz"))

        # Merge with remaining annnotations
        for cl in range(2, len(total_channels)):
            _label = load(join(path, eid, "segmentations", total_channels["class"][cl] + ".nii.gz"))
            label[_label == 1] = cl

        # Save as new label
        label.meta["filename_or_obj"] = "annotation.nii.gz"
        Save(join(path, eid))(label)


if __name__ == "__main__":

    main(sys.argv[1])
