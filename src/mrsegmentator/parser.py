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
import os


def initialize():
    name = "MRSegmentator"
    desc = "Robust Multi-Modality Segmentation of 40 Classes in MRI and CT Sequences"
    epilog = "Charité AG KI - 2024"

    parser = argparse.ArgumentParser(prog=name, description=desc, epilog=epilog)

    parser.add_argument("--modeldir", type=str, required=True, help="model directory")
    parser.add_argument("--outdir", type=str, required=True, help="output directory")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input image or directory with nifti images",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(5),
        help="choose a model based on the validation folds",
    )
    parser.add_argument("--crossval", action="store_true", help="run each model individually")

    parser.add_argument(
        "--force_LPS",
        action="store_true",
        help="change image orientation to LPS. Segmentations will be stored on the images' original orientation. (requires more memory)",  # noqa: E501
    )
    parser.add_argument("--postfix", type=str, default="seg", help="postfix")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("--cpu_only", action="store_true", help="don't use a gpu")

    args = parser.parse_args()
    return args


def assert_namespace(namespace):
    # requirements
    assert os.path.isdir(namespace.modeldir), f"Model directory {namespace.modeldir} not found"
    assert os.path.isdir(namespace.outdir), f"Output directory {namespace.outdir} not found"
    assert os.path.isfile(namespace.input) or os.path.isdir(
        namespace.input
    ), f"Input {namespace.input} not found"
