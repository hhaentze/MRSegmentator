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
from pathlib import Path
from typing import List, NoReturn, Tuple

import numpy as np


# Yield successive n-sized
# chunks from l.
def divide_chunks(l: List, n: int):  # noqa: E741
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def add_postfix(name: str, postfix: str):
    if Path(name).suffix == ".nii":
        return Path(name).stem + "_" + postfix + ".nii"
    elif Path(name).suffix == ".gz":
        return add_postfix(name[:-3], postfix) + ".gz"
    else:
        raise ValueError("Files must end with either .nii or .nii.gz")


def split_image(img: np.ndarray, margin=2) -> Tuple[np.ndarray, dict]:
    assert img.ndim == 4, f"Unexpected number of dimensions: {img.ndim}"
    depth = img.shape[1]
    img1 = img[:, : depth // 2 + margin, :, :]
    img2 = img[:, depth // 2 - margin :, :, :]
    return img1, img2


def stitch_segmentations(seg1: np.ndarray, seg2: np.ndarray, margin=2) -> np.ndarray:
    assert (
        seg1.ndim == 3 and seg2.ndim == 3
    ), f"Unexpected number of dimensions: {seg1.ndim} and {seg2.ndim}"

    # delete margin
    seg1 = seg1[:-margin, :, :]
    seg2 = seg2[margin:, :, :]

    # concatenate
    seg_combined = np.concatenate([seg1, seg2], axis=0)

    return seg_combined


def flatten(xss: List[List]) -> List:
    return [x for xs in xss for x in xs]


def disable_nnunet_path_warnings() -> NoReturn:
    """disable warning message about undefined environmental variables
    (We assign temporary arbitrary values. The script does not use these)"""

    if os.environ.get("nnUNet_raw") is None:
        os.environ["nnUNet_raw"] = "empty"
    if os.environ.get("nnUNet_preprocessed") is None:
        os.environ["nnUNet_preprocessed"] = "empty"
    if os.environ.get("nnUNet_results") is None:
        os.environ["nnUNet_results"] = "empty"


disable_nnunet_path_warnings()
