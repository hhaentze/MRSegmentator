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

from pathlib import Path
from typing import List


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
