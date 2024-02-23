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
