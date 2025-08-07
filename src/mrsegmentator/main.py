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

import time
from datetime import timedelta
from os.path import basename, join

from dicom_helper import utils as dcm_utils
from dicom_helper.dicom_conversion import dicom_to_nifti as d2n
from dicom_helper.dicom_conversion import nifti_to_dicom_seg as n2seg
from mrsegmentator import config, parser, utils

config.disable_nnunet_path_warnings()

from mrsegmentator.inference import infer  # noqa: E402


def main() -> None:

    # initialize Parser
    namespace = parser.initialize()
    parser.assert_namespace(namespace)

    try:
        images = utils.read_images(namespace.input)
        IS_DICOM = False

    except FileNotFoundError as e:
        # mabye the user specified a DICOM directory
        if dcm_utils.has_dicom_file(namespace.input):
            IS_DICOM = True
            target_name = basename(namespace.input) + f"_{namespace.postfix}"
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

    start_time = time.time()
    # run inference
    infer(
        images,
        namespace.outdir,
        folds,
        namespace.postfix,
        namespace.split_level,
        namespace.verbose,
        namespace.cpu_only,
        namespace.batchsize,
        namespace.nproc,
        namespace.nproc_export,
        namespace.split_margin,
    )
    end_time = time.time()
    time_delta = timedelta(seconds=round(end_time - start_time))
    print(f"Finished segmentation in {time_delta}.")

    if IS_DICOM:
        print("Converting segmentation to DICOM SEG")
        n2seg(
            nifti_path=join(namespace.outdir, target_name + ".nii.gz"),
            template_dir=namespace.input,
            output_file=join(namespace.outdir, target_name + ".dcm"),
        )


if __name__ == "__main__":
    main()
