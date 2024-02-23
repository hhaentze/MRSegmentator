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

import ntpath
from typing import List, Tuple, Union

import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from mrsegmentator.simpleitk_reader_writer import SimpleITKIO
from mrsegmentator.utils import add_postfix, divide_chunks


def infer(
    model_dir: str,
    folds: Union[List[int], Tuple[int, ...]],
    outdir: str,
    images: List[str],
    postfix: str = "seg",
    force_LPS: bool = False,
    verbose: bool = False,
    cpu_only: bool = False,
):
    """Run model to create segmentations
    model_dir: path to model directory
    folds: which models to use for inference
    outdir: path to output directory
    images: list with paths to images
    force_LPS: change orientation to LPS before inference"""

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

    if not force_LPS:
        # paths to output images
        image_names = [ntpath.basename(f) for f in images]
        out_names = [add_postfix(name, postfix) for name in image_names]

        # variant 1, use list of files as inputs
        predictor.predict_from_files(
            [[f] for f in images],
            [join(outdir, f) for f in out_names],
            save_probabilities=False,
            overwrite=False,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

    else:
        # load batch of images
        chunk_size = 4
        for img_chunk in divide_chunks(images, chunk_size):
            np_chunk = [SimpleITKIO().read_images([f], force_LPS=True) for f in img_chunk]
            imgs = [f[0] for f in np_chunk]
            props = [f[1] for f in np_chunk]

            # inference
            segmentations = predictor.predict_from_list_of_npy_arrays(
                imgs,
                None,
                props,
                None,
                2,
                save_probabilities=False,
                num_processes_segmentation_export=2,
            )

            # paths to output images
            image_names = [ntpath.basename(f) for f in img_chunk]
            out_names = [add_postfix(name, postfix) for name in image_names]

            # save images
            for seg, p, out in zip(segmentations, props, out_names):
                SimpleITKIO().write_seg(seg, join(outdir, out), p, force_LPS=True)
