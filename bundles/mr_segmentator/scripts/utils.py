import logging
import multiprocessing
import os
import resource
from pathlib import Path
from typing import Callable, Dict, OrderedDict, Union

import monai
import pandas as pd
import psutil
import torch

logger = logging.getLogger(__name__)

USE_AMP = monai.utils.get_torch_version_tuple() >= (1, 6)  # type: ignore


def num_workers() -> int:
    """Get max supported workers -2 for multiprocessing"""

    n_workers = multiprocessing.cpu_count() - 2  # leave two workers so machine can still respond

    # check if we will run into OOM errors because of too many workers
    # In most projects 2-4GB/Worker seems to be save
    available_ram_in_gb = psutil.virtual_memory()[0] / 1021**3
    max_workers = int(available_ram_in_gb // 4)
    if max_workers < n_workers:
        n_workers = max_workers

    # now check for max number of open files allowed on system
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    # giving each worker at least 216 open processes should allow them to run smoothly
    max_workers = soft_limit // 216

    if max_workers < n_workers:
        logger.info(
            "Number of allowed open files is to small, "
            "which might lead to problems with multiprocessing"
            "Current limits are:\n"
            f"\t soft_limit: {soft_limit}\n"
            f"\t hard_limit: {hard_limit}\n"
            "try increasing the limits to at least {216*n_workers}."
            "See https://superuser.com/questions/1200539/cannot-increase-open-file-limit-past"
            "-4096-ubuntu for more details.\n"
            "Will use torch.multiprocessing.set_sharing_strategy('file_system') as a workarround."
        )
        n_workers = min(32, n_workers)
        torch.multiprocessing.set_sharing_strategy("file_system")
    logger.info(f"using number of workers: {n_workers}")

    return n_workers


def get_datalist(df: pd.DataFrame, image_path: str, label_path: str) -> Dict[str, str]:
    df["image"] = image_path + df["image"]
    df["label"] = label_path + df["label"]
    data_list = df.to_dict("records")
    return data_list


IMAGE_FILES = [".nii", ".nii.gz", ".nrrd", ".dcm"]


def parse_data_for_inference(fn_or_dir: str = None, recursion_depth: int = 1) -> Union[None, Dict]:
    """Convert filepath to data_dict"""

    if not fn_or_dir:
        return

    if os.path.isfile(fn_or_dir):
        data_dict = [{"image": fn_or_dir}]

    elif os.path.isdir(fn_or_dir):
        data_dict = []
        for fn in os.listdir(fn_or_dir):
            _fn = os.path.join(fn_or_dir, fn)
            if any([_fn.endswith(ext) for ext in IMAGE_FILES]):
                data_dict += [{"image": _fn}]
            elif os.path.isdir(_fn) and recursion_depth > 0:
                data_dict += parse_data_for_inference(_fn, recursion_depth - 1)
    else:
        raise FileNotFoundError(fn_or_dir)

    return data_dict


def get_meta_dict(image_key) -> Callable:
    def _inner(batch) -> list:
        """Get dict of metadata from engine. Needed as `batch_transform` for MetricSaver to also save filenames"""
        key = image_key[0] if isinstance(image_key, list) else image_key
        return [item[key].meta for item in batch]

    return _inner


def adapt_filename(x):
    file = Path(x.meta["filename_or_obj"])
    seq = str(file.parent.name)
    idx = str(file.parent.parent.name)
    x.meta["filename_or_obj"] = idx + "_" + seq
    return x


def adapt_filename_ukbb(x):
    file = Path(x.meta["filename_or_obj"])
    idx = str(file.parent.name)
    x.meta["filename_or_obj"] = idx + "_" + file.name
    return x


def maybe_load_checkpoint(
    model: torch.nn.Module, checkpoint_path: OrderedDict, network_key: str = None
) -> torch.nn.Module:
    """Loads weights from checkpoint, even if some keys do not match.
    Should the checkpoint not exist, it will not break the skript.
    """
    # In theory, monai.handlers.CheckPointLoader with strict_shape should do
    # the same, but somehow it does not load any weights for me. It was
    # faster to implement this function than to debug the CheckPointLoader
    if not os.path.exists(checkpoint_path):
        logger.warn(f"No valid checkpoint at {checkpoint_path}. Returning model as is.")
        return model
    src_state_dict = torch.load(checkpoint_path)
    if network_key:
        src_state_dict = src_state_dict[network_key]
    dst_state_dict = model.state_dict()
    non_matched_keys = []
    matched_keys = []
    for k, v in src_state_dict.items():
        if k in dst_state_dict.keys():
            if dst_state_dict[k].shape == v.shape:
                dst_state_dict[k] = v
                matched_keys.append(k)
            else:
                logger.debug(f"Key {k} did not match.")
                non_matched_keys.append(k)
        else:
            logger.debug(f"Key {k} did not found.")
            non_matched_keys.append(k)
    logger.info(f"Matched {len(matched_keys)}/{len(matched_keys) + len(non_matched_keys)} keys.")
    if len(non_matched_keys) < 5:
        logger.info(f"The following keys did not match: {non_matched_keys}")
    model.load_state_dict(dst_state_dict, strict=False)

    return model
