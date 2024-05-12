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

import json
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import NoReturn

from tqdm import tqdm

logger = logging.getLogger(__name__)
WEIGHTS_VERSION = 1.0
WEIGHTS_URL = "https://github.com/hhaentze/MRSegmentator/releases/download/v1.0.0/weights.zip"


def get_weights_dir():

    if "MRSEG_WEIGHTS_PATH" in os.environ:
        weights_dir = Path(os.environ["MRSEG_WEIGHTS_PATH"])
    else:
        module_dir = Path(os.path.dirname(__file__))
        weights_dir = module_dir / "weights"

    weights_dir.mkdir(exist_ok=True)
    print(weights_dir)

    return weights_dir


def read_config():

    weights_dir = get_weights_dir()

    if os.path.exists(weights_dir / "version.json"):
        with open(weights_dir / "version.json", "r") as f:
            config_info = json.load(f)

        return config_info

    else:
        return {"weights_version": 0.0}


def disable_nnunet_path_warnings() -> NoReturn:
    """disable warning message about undefined environmental variables
    (We assign temporary arbitrary values. The script does not use these)"""

    if os.environ.get("nnUNet_raw") is None:
        os.environ["nnUNet_raw"] = "empty"
    if os.environ.get("nnUNet_preprocessed") is None:
        os.environ["nnUNet_preprocessed"] = "empty"
    if os.environ.get("nnUNet_results") is None:
        os.environ["nnUNet_results"] = "empty"


def user_guard(func):
    """Check for user defined environment variables. We do NOT want to change user directories"""

    if "MRSEG_WEIGHTS_PATH" in os.environ:
        logger.info("User defined environment variables detected, skip directory operations.")
        return

    else:
        return func


@user_guard
def delete_weights():

    return
    weights_dir = get_weights_dir()
    if os.path.isdir(weights_dir):
        shutil.rmtree(weights_dir)


@user_guard
def download_weights():

    weights_dir = get_weights_dir()

    print("Downloading pretrained weights...")
    # Retrieve file size
    with urllib.request.urlopen(WEIGHTS_URL) as response:
        file_size = int(response.info().get("Content-Length", -1))

    with tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=WEIGHTS_URL.split("/")[-1],
    ) as pbar:

        def update_progress(block_num, block_size, total_size):
            if pbar.total != total_size:
                pbar.total = total_size
            pbar.update(block_num * block_size - pbar.n)

        # Download the file
        urllib.request.urlretrieve(
            WEIGHTS_URL, weights_dir / "mrsegmentator_weights.zip", reporthook=update_progress
        )

    print("Extacting pretrained weights...")
    with zipfile.ZipFile(weights_dir / "mrsegmentator_weights.zip", "r") as zip_ref:
        zip_ref.extractall(weights_dir)

    os.remove(weights_dir / "mrsegmentator_weights.zip")


def setup_mrseg():

    weights_dir = get_weights_dir()

    # Check if weights are downloaded
    if not os.path.exists(weights_dir):
        download_weights()

    # Check if weights are up to date
    else:
        config_info = read_config()
        if config_info["weights_version"] < WEIGHTS_VERSION:
            print(
                f"A new version ({WEIGHTS_VERSION}) of weights was found. "
                + f"You have version {config_info['weights_version']}."
            )
            delete_weights()
            download_weights()

    disable_nnunet_path_warnings()

    return weights_dir
