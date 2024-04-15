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

import os

from batchgenerators.utilities.file_and_folder_operations import join

from mrsegmentator import parser
from mrsegmentator.inference import infer


def crossval(namespace, images):
    """Run each model individually"""

    for fold in range(5):
        # make directory
        outdir = join(namespace.outdir, "fold" + str(fold))
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        # run inference
        infer(
            namespace.modeldir,
            (fold,),
            outdir,
            images,
            namespace.postfix,
            namespace.is_LPS,
            namespace.split_level,
            namespace.verbose,
            namespace.cpu_only,
            namespace.batchsize,
            namespace.nproc,
            namespace.nproc_export,
        )


def read_images(namespace):
    # images must be of nifti format
    condition = lambda x: x[-7:] == ".nii.gz" or x[-4:] == ".nii"

    # look for images in input directory
    if os.path.isdir(namespace.input):
        images = [f.path for f in os.scandir(namespace.input) if condition(f.name)]
        assert (
            len(images) > 0
        ), f"no images with file ending .nii or .nii.gz in direcotry {namespace.input}"
    else:
        images = [namespace.input]
        assert condition(images[0]), f"file ending of {namespace.input} neither .nii nor .nii.gz"

    return images


def main():
    # initialize Parser
    namespace = parser.initialize()
    parser.assert_namespace(namespace)

    # select images for segmentation
    images = read_images(namespace)

    # run all models individually
    if namespace.crossval:
        crossval(namespace, images)
        return

    # ensemble prediction
    if namespace.fold is None:
        folds = (
            0,
            1,
            2,
            3,
            4,
        )

    # single prediction
    else:
        folds = (namespace.fold,)

    # run inference
    infer(
        namespace.modeldir,
        folds,
        namespace.outdir,
        images,
        namespace.postfix,
        namespace.is_LPS,
        namespace.split_level,
        namespace.verbose,
        namespace.cpu_only,
        namespace.batchsize,
        namespace.nproc,
        namespace.nproc_export,
    )


if __name__ == "__main__":
    main()
