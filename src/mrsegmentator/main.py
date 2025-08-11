# Copyright 2024-2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import time
from datetime import timedelta
from os.path import basename, join

import torch

from dicom_helper import utils as dcm_utils
from dicom_helper.dicom_conversion import dicom_to_nifti as d2n
from dicom_helper.dicom_conversion import nifti_to_dicom_seg as n2seg
from mrsegmentator import config, parser, utils
from mrsegmentator.custom_logger import setup_logging

config.disable_nnunet_path_warnings()

from mrsegmentator.inference import infer  # noqa: E402


def main() -> None:

    # initialize Parser
    namespace = parser.initialize()
    parser.assert_namespace(namespace)

    setup_logging(level=namespace.log_level)

    try:
        images = utils.read_images(namespace.input)
        IS_DICOM = False

    except FileNotFoundError as e:
        logging.debug("No images found in input directory. Checking if input is a DICOM directory.")
        if dcm_utils.has_dicom_file(namespace.input):
            logging.info("DICOM found. Converting to NIfTI...")
            IS_DICOM = True
            target_name = basename(namespace.input)
            if target_name == ".":
                target_name = "dicom_input"
            d2n(
                dicom_dir=namespace.input,
                output_dir=namespace.outdir,
                output_name=target_name + ".nii.gz",
            )
            images = utils.read_images(join(namespace.outdir, target_name + ".nii.gz"))
        else:
            raise e

    # ensemble/single prediction
    if namespace.fold is None:
        folds = (0, 1, 2, 3, 4)
    else:
        folds = (namespace.fold,)  # type: ignore

    # check gpu availability
    if not namespace.cpu_only and not torch.cuda.is_available():
        logging.warning("No GPU available, running on CPU.")
        if len(folds) > 1:
            logging.warning(
                "Running inference with multiple folds on CPU will be slow. "
                "We recommend setting --fold 0 to deactivate ensembling."
            )
        namespace.cpu_only = True

    start_time = time.time()
    infer(
        images,
        namespace.outdir,
        folds,
        namespace.postfix,
        namespace.split_level,
        True if namespace.log_level == "DEBUG" else False,
        namespace.cpu_only,
        namespace.batchsize,
        namespace.nproc,
        namespace.nproc_export,
        namespace.split_margin,
        not namespace.no_tqdm,
    )
    end_time = time.time()
    time_delta = timedelta(seconds=round(end_time - start_time))
    logging.info(f"Finished segmentation in {time_delta}.")

    if IS_DICOM:
        logging.info("Converting segmentation to DICOM SEG")
        n2seg(
            nifti_path=join(namespace.outdir, target_name + f"_{namespace.postfix}.nii.gz"),
            template_dir=namespace.input,
            output_file=join(namespace.outdir, target_name + f"_{namespace.postfix}.dcm"),
        )


if __name__ == "__main__":
    main()
