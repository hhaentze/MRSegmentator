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

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def initialize():
    name = "MRSegmentator"
    desc = "Robust Multi-Modality Segmentation of 40 Classes in MRI and CT Sequences"
    epilog = "Charité AG KI - 2024"

    parser = argparse.ArgumentParser(prog=name, description=desc, epilog=epilog)

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input image or directory with nifti images",
    )

    parser.add_argument("--outdir", type=str, default="segmentations", help="output directory")

    parser.add_argument(
        "--fold",
        type=int,
        choices=range(5),
        help="choose a model based on the validation folds",
    )

    parser.add_argument(
        "--batchsize",
        type=int,
        default=8,
        help="how many images can be loaded to memory at the same time, ideally this should equal the dataset size",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=3,
        help="number of processes",
    )
    parser.add_argument(
        "--nproc_export",
        type=int,
        default=8,
        help="number of processes for exporting the segmentations",
    )
    parser.add_argument(
        "--split_level",
        type=int,
        default=0,
        help="split images to reduce memory usage. Images are split recusively: A split level of x will produce 2^x smaller images",  # noqa: E501
    )
    parser.add_argument("--postfix", type=str, default="seg", help="postfix")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("--cpu_only", action="store_true", help="don't use a gpu")

    args = parser.parse_args()
    return args


def assert_namespace(namespace):
    # requirements
    if namespace.outdir != "segmentations":
        assert os.path.isdir(namespace.outdir), f"Output directory {namespace.outdir} not found"
    assert os.path.isfile(namespace.input) or os.path.isdir(
        namespace.input
    ), f"Input {namespace.input} not found"

    # constraints
    assert namespace.batchsize >= 1, "batchsize must be greater than 1"
    assert namespace.nproc >= 1, "number of processes must be greater than 1"
    assert (
        namespace.nproc_export >= 1
    ), "number of processes for image export must be greater than 1"
    assert namespace.split_level >= 0, "split level must be equal or greather than zero"

    # warnings
    if namespace.split_level >= 3:
        logger.warning(
            f"Warning: Based on the specified split level of {namespace.split_level} images will be cut into 2^{namespace.split_level}={pow(2,namespace.split_level)} smaller images. "  # noqa: E501
            + "Are you sure this is intended?"
        )
