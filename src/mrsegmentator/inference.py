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

from mrsegmentator import utils  # isort:skip
from mrsegmentator.simpleitk_reader_writer import SimpleITKIO  # isort:skip

import ntpath
from typing import List, NoReturn, Tuple, Union

import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def infer(
    model_dir: str,
    outdir: str,
    images: List[str],
    folds: Union[List[int], Tuple[int, ...]],
    postfix: str = "seg",
    is_LPS: bool = False,
    split_level: int = 0,
    verbose: bool = False,
    cpu_only: bool = False,
    batchsize: int = 3,
    nproc: int = 3,
    nproc_export: int = 8,
) -> NoReturn:
    """Run model to create segmentations
    model_dir: path to model directory
    folds: which models to use for inference
    outdir: path to output directory
    images: list with paths to images
    is_LPS: do not change orientation to LPS before inference"""

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device("cpu") if cpu_only else torch.device("cuda", 0),
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=True,
    )

    # initialize the network architecture, load the checkpoints
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )

    if split_level == 0 and is_LPS:
        # paths to output images
        image_names = [ntpath.basename(f) for f in images]
        out_names = [utils.add_postfix(name, postfix) for name in image_names]

        # variant 1, use list of files as inputs
        predictor.predict_from_files(
            [[f] for f in images],
            [join(outdir, f) for f in out_names],
            save_probabilities=False,
            overwrite=False,
            num_processes_preprocessing=nproc,
            num_processes_segmentation_export=nproc_export,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

    elif split_level == 0 and not is_LPS:
        # load batch of images
        # (Loading all images at once might require too much memory, instead we procede chunk wise)
        chunk_size = batchsize
        for i, img_chunk in enumerate(utils.divide_chunks(images, chunk_size)):

            print(
                f"Processing image { chunk_size*i + 1 } to {chunk_size*i + len(img_chunk)} out of {len(images)} images."
            )

            # load images
            np_chunk = [SimpleITKIO().read_image(f, is_LPS=is_LPS, verbose=True) for f in img_chunk]
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
                SimpleITKIO().write_seg(seg, join(outdir, out), p, is_LPS=is_LPS, verbose=True)

    else:
        # sequential inference (parallelization would increase memory)
        for i, img in enumerate(images):

            # load image
            print(f"Processing image { i + 1 } out of {len(images)} images.")
            np_img, prop = SimpleITKIO().read_image(img, is_LPS=is_LPS, verbose=True)

            # split image to reduce memory usage
            np_imgs = [np_img]
            for _ in range(split_level):
                np_imgs = utils.flatten([utils.split_image(n) for n in np_imgs])

            # infer
            segmentations = []
            for n in np_imgs:
                seg = predictor.predict_single_npy_array(n, prop, None, None, False)
                segmentations += [seg]

            # stitch segmentations back together
            for _ in range(split_level):
                segmentations = [
                    utils.stitch_segmentations(segmentations[_i], segmentations[_i + 1])
                    for _i in range(0, len(segmentations), 2)
                ]

            # paths to output image
            out_name = utils.add_postfix(ntpath.basename(img), postfix)

            # save image
            SimpleITKIO().write_seg(
                segmentations[0], join(outdir, out_name), prop, is_LPS=is_LPS, verbose=True
            )
