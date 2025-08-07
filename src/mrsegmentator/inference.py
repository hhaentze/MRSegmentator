# Copyright 2024-2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import ntpath
from pathlib import Path
from typing import List, Tuple, Union

import torch

from mrsegmentator import config, utils
from mrsegmentator.simpleitk_reader_writer import SimpleITKIO

config.disable_nnunet_path_warnings()
from batchgenerators.utilities.file_and_folder_operations import join  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402

logger = logging.getLogger(__name__)


def infer(
    images: List[str],
    outdir: str,
    folds: Union[List[int], Tuple[int, ...]] = [0, 1, 2, 3, 4],
    postfix: str = "seg",
    split_level: int = 0,
    verbose: bool = False,
    cpu_only: bool = False,
    batchsize: int = 3,
    nproc: int = 3,
    nproc_export: int = 8,
    split_margin: int = 3,
    allow_tqdm=True,
) -> None:
    """Run model to create segmentations
    folds: which models to use for inference
    outdir: path to output directory
    images: list with paths to images
    postfix: default='seg'
    split_level: split images to reduce memory footprint
    split_margin: Images are splitted with an overlap of 2xmargin, to avoid hard differences
                  between segmentations of top and bottom image.
    """

    # initialize weights directory
    config.setup_mrseg()

    # make output directory
    Path(outdir).mkdir(exist_ok=True)

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device("cpu") if cpu_only else torch.device("cuda", 0),
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=allow_tqdm,
    )

    # initialize the network architecture, load the checkpoints
    predictor.initialize_from_trained_model_folder(
        config.get_weights_dir(),
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )

    if split_level == 0:

        # load batch of images
        # (loading all images at once might require too much memory, instead we procede chunk wise)
        for i, img_chunk in enumerate(utils.divide_chunks(images, batchsize)):

            logger.info(
                f"Processing image { batchsize*i + 1 } to {batchsize*i + len(img_chunk)} out of {len(images)} images."
            )

            # load images
            np_chunk = [SimpleITKIO().read_image(f, verbose=True) for f in img_chunk]
            imgs = [f[0] for f in np_chunk]
            props = [f[1] for f in np_chunk]

            # inference
            segmentations = predictor.predict_from_list_of_npy_arrays(
                imgs,
                None,
                props,
                None,
                num_processes=nproc,
                save_probabilities=False,
                num_processes_segmentation_export=nproc_export,
            )

            # paths to output images
            image_names = [ntpath.basename(f) for f in img_chunk]
            out_names = [utils.add_postfix(name, postfix) for name in image_names]

            # save images
            for seg, p, out in zip(segmentations, props, out_names):
                SimpleITKIO().write_seg(seg, join(outdir, out), p, verbose=True)

    else:
        # sequential inference (parallelization would increase memory)
        for i, img in enumerate(images):

            # load image
            logger.info(f"Processing image { i + 1 } out of {len(images)} images.")
            np_img, prop = SimpleITKIO().read_image(img, verbose=True)

            # split image to reduce memory usage
            np_imgs = [np_img]
            for _ in range(split_level):
                np_imgs = utils.flatten(
                    [utils.split_image(n, margin=split_margin) for n in np_imgs]
                )
            logger.info(f"Splitted image into {len(np_imgs)} volumes of size {np_imgs[0].shape}.")

            # infer
            segmentations = []
            for n in np_imgs:
                seg = predictor.predict_single_npy_array(n, prop, None, None, False)
                segmentations += [seg]

            # stitch segmentations back together
            for _ in range(split_level):
                segmentations = [
                    utils.stitch_segmentations(
                        segmentations[_i], segmentations[_i + 1], margin=split_margin
                    )
                    for _i in range(0, len(segmentations), 2)
                ]

            # paths to output image
            out_name = utils.add_postfix(ntpath.basename(img), postfix)

            # save image
            SimpleITKIO().write_seg(segmentations[0], join(outdir, out_name), prop, verbose=True)
