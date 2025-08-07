# Copyright 2024-2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import logging
from typing import Any, Dict, Tuple

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


def get_orientation_string(image) -> str:
    direction = np.array(image.GetDirection()).reshape(3, 3)

    # Because SimpleITK uses LPS, we need to map it to RAS for consistency with nibabel
    # L (Left, +X) -> R (Right, -X)
    # P (Posterior, +Y) -> A (Anterior, -Y)
    # S (Superior, +Z) stays S in both
    # TODO evaluate if this is actually correct

    # Mapping LPS to RAS axis labels
    ras_labels = np.array(
        [
            ["R", "L"],  # X: Right (−), Left (+)
            ["A", "P"],  # Y: Anterior (−), Posterior (+)
            ["S", "I"],
        ]
    )  # Z: Superior (+), Inferior (−)

    orientation = ""
    for axis in direction.T:
        dominant = np.argmax(np.abs(axis))
        sign = np.sign(axis[dominant])

        # Flip sign for X and Y to convert from LPS to RAS
        if dominant in [0, 1]:  # X or Y
            sign *= -1

        orientation += ras_labels[dominant, 0 if sign > 0 else 1]

    return orientation


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
        orientation = get_orientation_string(itk_image)
        itk_image = sitk.DICOMOrient(itk_image, "LPS")

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
