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

import logging
from typing import Any, Dict, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def log(verbose: bool, message: str) -> None:
    """Log a message based on the verbosity level."""
    if verbose:
        logger.info(message)
    else:
        logger.debug(message)


class SimpleITKIO:

    def read_image(
        self,
        image_fname: str,
        verbose: bool = False,
    ) -> Tuple[NDArray, Dict[str, Any]]:

        log(verbose, f"Read {image_fname}")

        # read image and save meta data
        itk_image = sitk.ReadImage(image_fname)
        spacing = itk_image.GetSpacing()
        origin = itk_image.GetOrigin()
        direction = itk_image.GetDirection()
        itk_image = sitk.DICOMOrient(itk_image, "LPS")

        nib_image = nib.load(image_fname)
        orientation = "".join(nib.aff2axcodes(nib_image.affine))  # type: ignore

        # transform image to numpy array
        npy_image = sitk.GetArrayFromImage(itk_image)
        assert (
            npy_image.ndim == 3
        ), f"Unexpected number of dimensions: {npy_image.ndim} in file {image_fname}"
        npy_image = npy_image[None]

        # combine numpy array with meta data
        _dict = {
            "sitk_stuff": {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                "spacing": spacing,
                "origin": origin,
                "direction": direction,
                "orientation": orientation,
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            "spacing": list(spacing)[::-1],
        }

        log(verbose, f"Image Size: {npy_image.shape}\tSpacing: {_dict['spacing']}")

        return npy_image.astype(np.float32), _dict

    def read_seg(self, seg_fname: str) -> Tuple[NDArray, Dict[str, Any]]:
        return self.read_image(seg_fname)

    def write_seg(
        self,
        seg: NDArray,
        output_fname: str,
        properties: Dict[str, Any],
        verbose: bool = False,
    ) -> None:

        assert (
            seg.ndim == 3
        ), "segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y"
        log(verbose, f"Write {output_fname}")

        output_dimension = len(properties["sitk_stuff"]["spacing"])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8))
        itk_image = sitk.DICOMOrient(itk_image, properties["sitk_stuff"]["orientation"])
        itk_image.SetSpacing(properties["sitk_stuff"]["spacing"])
        itk_image.SetOrigin(properties["sitk_stuff"]["origin"])
        itk_image.SetDirection(properties["sitk_stuff"]["direction"])

        sitk.WriteImage(itk_image, output_fname, True)
