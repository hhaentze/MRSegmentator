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
from typing import Any, Callable, Iterator, List, Tuple

import numpy as np
from numpy.typing import NDArray


def read_images(namespace: Any) -> List[str]:
    # images must be of nifti format
    condition: Callable[[str], bool] = lambda x: x[-7:] == ".nii.gz" or x[-4:] == ".nii"

    # look for images in input directory
    if os.path.isdir(namespace.input):
        images = [f.path for f in os.scandir(namespace.input) if condition(f.name)]
        assert (
            len(images) > 0
        ), f"no images with file ending .nii or .nii.gz in direcotry {namespace.input}"
    else:
        images = [namespace.input]
        assert condition(images[0]), f"file ending of {namespace.input} neither .nii nor .nii.gz"

    return images


# Yield successive n-sized
# chunks from l.
def divide_chunks(l: List, n: int) -> Iterator[List]:  # noqa: E741
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def add_postfix(name: str, postfix: str) -> str:
    if Path(name).suffix == ".nii":
        return Path(name).stem + "_" + postfix + ".nii"
    elif Path(name).suffix == ".gz":
        return add_postfix(name[:-3], postfix) + ".gz"
    else:
        raise ValueError("Files must end with either .nii or .nii.gz")


def split_image(img: NDArray, margin: int = 2) -> Tuple[NDArray, NDArray]:
    assert img.ndim == 4, f"Unexpected number of dimensions: {img.ndim}"
    depth = img.shape[1]
    img1 = img[:, : depth // 2 + margin, :, :]
    img2 = img[:, depth // 2 - margin :, :, :]
    return img1, img2


def stitch_segmentations(seg1: NDArray, seg2: NDArray, margin: int = 2) -> NDArray:
    assert (
        seg1.ndim == 3 and seg2.ndim == 3
    ), f"Unexpected number of dimensions: {seg1.ndim} and {seg2.ndim}"

    # delete margin
    seg1 = seg1[:-margin, :, :]
    seg2 = seg2[margin:, :, :]

    # concatenate
    seg_combined = np.concatenate([seg1, seg2], axis=0)

    return seg_combined


def flatten(xss):
    return [x for xs in xss for x in xs]
