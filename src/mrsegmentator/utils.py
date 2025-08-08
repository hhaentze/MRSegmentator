# Copyright 2024-2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import os
from typing import Iterator, List, Tuple

import numpy as np
from numpy.typing import NDArray

SUPPORTED_FILES = (".nii", ".nii.gz", ".mha", ".nrrd")


def is_supported(file: str) -> bool:
    for ext in SUPPORTED_FILES:
        if file.endswith(ext):
            return True
    return False


def read_images(input: str) -> List[str]:
    if os.path.isdir(input):
        images = [f.path for f in os.scandir(input) if is_supported(f.name)]
        if len(images) == 0:
            raise FileNotFoundError(
                f"No images with supported file endings {SUPPORTED_FILES} in directory {input}"
            )
    else:
        if not is_supported(input):
            raise ValueError(
                f"File {input} not supported. (Supported file types: {SUPPORTED_FILES})"
            )
        images = [input]

    return images


# Yield successive n-sized chunks from l.
def divide_chunks(l: List, n: int) -> Iterator[List]:  # noqa: E741
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def add_postfix(name: str, postfix: str) -> str:
    for ext in SUPPORTED_FILES:
        if name.endswith(ext):
            base = name[: -len(ext)]
            return f"{base}_{postfix}{ext}"
    raise ValueError(f"File must end with one of: {SUPPORTED_FILES}")


def split_image(img: NDArray, margin: int = 3) -> Tuple[NDArray, NDArray]:
    assert img.ndim == 4, f"Unexpected number of dimensions: {img.ndim}"
    depth = img.shape[1]
    img1 = img[:, : depth // 2 + margin, :, :]
    img2 = img[:, depth // 2 - margin :, :, :]
    return img1, img2


def stitch_segmentations(seg1: NDArray, seg2: NDArray, margin: int = 3) -> NDArray:
    assert (
        seg1.ndim == 3 and seg2.ndim == 3
    ), f"Unexpected number of dimensions: {seg1.ndim} and {seg2.ndim}"

    # delete margin
    if margin > 0:
        seg1 = seg1[:-margin, :, :]
        seg2 = seg2[margin:, :, :]

    # concatenate
    seg_combined = np.concatenate([seg1, seg2], axis=0)

    return seg_combined


def flatten(xss):
    return [x for xs in xss for x in xs]
