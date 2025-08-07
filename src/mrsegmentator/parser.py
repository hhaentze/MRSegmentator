# Copyright 2024-2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import argparse
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def initialize() -> Any:
    name = "MRSegmentator"
    desc = "Multi-Modality Segmentation of 40 Classes in MRI and CT"
    epilog = (
        "AIAH Lab – 2024\n\n"
        "Group website: https://ai-assisted-healthcare.com\n"
        "Published paper: https://doi.org/10.1148/ryai.240777"
    )

    parser = argparse.ArgumentParser(
        prog=name,
        description=desc,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

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
    parser.add_argument(
        "--split_margin",
        type=int,
        default=3,
        help="split images with an overlap of 2xmargin to avoid hard cutt-offs between segmentations of top and bottom image",  # noqa: E501
    )

    parser.add_argument("--postfix", type=str, default="seg", help="postfix")
    parser.add_argument("--cpu_only", action="store_true", help="don't use a gpu")

    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument("--no_tqdm", action="store_true", help="disable tqdm progress bars")

    args = parser.parse_args()
    return args


def assert_namespace(namespace: Any) -> None:
    # requirements

    assert os.path.isdir(
        Path(namespace.outdir).parent
    ), f"Parent of output directory {namespace.outdir} not found"
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
    assert namespace.split_margin >= 0, "split margin must be equal or greather than zero"

    # warnings
    if namespace.split_level >= 3:
        logger.warning(
            f"Warning: Based on the specified split level of {namespace.split_level} images will be cut into 2^{namespace.split_level}={pow(2,namespace.split_level)} smaller images. "  # noqa: E501
            + "Are you sure this is intended?"
        )
