"""
Stitching multiple sitk.Images to create one single large image
"""

from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm.autonotebook import tqdm


def left_upper(image1: sitk.Image, image2: sitk.Image) -> Tuple[int, int, int]:
    """
    Assuming that image1 and image2 are in the same coordinate system. This
    function returns the left upper point of the bunding box that encases
    the two images

    Args:
        image1 (sitk.Image): The first image
        image2 (sitk.Image): The secong image

    Returns:
        A tuple of the coordinates (x, y, z)
    """
    origin1 = image1.GetOrigin()
    origin2 = image2.GetOrigin()
    return tuple([min(o1, o2) for o1, o2 in zip(origin1, origin2)])


def get_spatial_size(image: sitk.Image) -> Tuple[int, int, int]:
    """
    While sitk.Image.GetSize returns the size of the pixel matrix,
    this function returns the spatial size of the image.

    Args:
        image (sitk.Image): The image to extract the spatial size

    Returns:
        tuple: A tuple with the spatial size in all dimensions
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    return tuple([sz * sp for sz, sp in zip(size, spacing)])


def right_lower(image1: sitk.Image, image2: sitk.Image) -> Tuple[int, int, int]:
    """
    Assuming that image1 and image2 are in the same coordinate system. This
    function returns the right lower point of the bunding box that encases
    the two images

    Args:
        image1 (sitk.Image): The first image
        image2 (sitk.Image): The secong image

    Returns:
        tuple: A tuple of the coordinates (x, y, z)
    """
    origin1 = image1.GetOrigin()
    origin2 = image2.GetOrigin()
    size1 = get_spatial_size(image1)
    size2 = get_spatial_size(image2)
    right_lower1 = [o + s for o, s in zip(origin1, size1)]
    right_lower2 = [o + s for o, s in zip(origin2, size2)]
    return tuple([max(r1, r2) for r1, r2 in zip(right_lower1, right_lower2)])


def resample_images_to_fit(
    fixed_image: sitk.Image, moving_image: sitk.Image
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Resamples both `moving_image` and `fixed_image`, so they occupy the same
    physical space. Useful as preprocessing before image registration

    o---o              o---------o    o---------o
    |im1| o---o        |         |    |         |
    |   | |im2|  = >   |   im1   |    |   im2   |
    o---o |   |        |         |    |         |
          o---o        o---------o    o---------o

    Args:
        fixed_image (sitk.Image): The first image
        moving_image (sitk.Image): The secong image

    Returns:
        tuple: A tuple of both images resampled.

    """
    origin_empty_image = left_upper(fixed_image, moving_image)
    right_lower_empty_image = right_lower(fixed_image, moving_image)
    spatial_size_emtpy_image = [
        abs(lu - rl) for lu, rl in zip(origin_empty_image, right_lower_empty_image)
    ]
    px_size_emtpy_image = [
        round(sz / sp) for sz, sp in zip(spatial_size_emtpy_image, fixed_image.GetSpacing())
    ]

    empty_image = sitk.Image(px_size_emtpy_image, fixed_image.GetPixelID())
    empty_image.SetSpacing(fixed_image.GetSpacing())
    empty_image.SetOrigin(origin_empty_image)
    empty_image.SetDirection(fixed_image.GetDirection())

    fixed_image = sitk.Resample(fixed_image, empty_image)
    moving_image = sitk.Resample(moving_image, empty_image)
    return fixed_image, moving_image


def translate_image(fixed_image: sitk.Image, moving_image: sitk.Image):
    """
    Translates a `moving_image` to align it with a `fixed_image` using the ElastixImageFilter.

    Args:
        fixed_image (sitk.Image): The fixed image that the `moving_image` will
            be aligned to.
        moving_image (sitk.Image): The image that will be translated to align
            with the `fixed_image`.

    Returns:
        tuple: A tuple containing the translated `moving_image` and the `ParameterMap`
            of the transformation applied.
    """
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("translation"))
    elastixImageFilter.Execute()
    # We need the translation parameter map, but rigid registration gives better results.
    parameter_map = elastixImageFilter.GetTransformParameterMap()[0].asdict()
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
    elastixImageFilter.Execute()
    translated_image = elastixImageFilter.GetResultImage()
    return translated_image, parameter_map


def combine_images(
    array: np.ndarray, fixed_image: sitk.Image, moving_image: sitk.Image, z_direction: str
) -> np.ndarray:
    """
    Parse `fixed_image` and `moving_image` onto an empty array.

    Args:
        array (np.ndarray): An (empty) array providing the common canvas to parse
            both images
        fixed_image (sitk.Image): The first image to parse onto the array
        moving_image (sitk.Image): The second image to parse onto the array
        z_direction (str): If caudal, `first_image` is treated as top image
            (caudal to `moving_image`)

    Returns:
        np.ndarray: The combined image as array
    """
    top = sitk.GetArrayFromImage(fixed_image if z_direction == "caudal" else moving_image)
    bottom = sitk.GetArrayFromImage(moving_image if z_direction == "caudal" else fixed_image)

    if top.shape == bottom.shape == array.shape:
        return np.stack([top, bottom], 0).max(0)

    # TODO: enable smoother pooling of the images
    fx, fy, fz = top.shape
    mx, my, mz = bottom.shape
    top = np.stack([top, array[-fx:, -fy:, -fz:]], 0).max(0)
    array[-fx:, -fy:, -fz:] = top
    bottom = np.stack([bottom, array[:mx, :my, :mz]], 0).max(0)
    array[:mx, :my, :mz] = bottom

    return array


def stitch_two_images(fixed_image: sitk.Image, moving_image: sitk.Image) -> sitk.Image:
    """
    Stitches two SimpleITK images into a single image by aligning `moving_image`
    to `fixed_image` and combining them.

    Parameters:
        fixed_image (sitk.Image): The fixed image that the `moving_image` will be
            aligned to.
        moving_image (sitk.Image): The image that will be translated to align with
            the `fixed_image`.

    Returns:
        sitk.Image: A single image resulting from combining `fixed_image` and
            `moving_image` after aligning `moving_image` to `fixed_image`.
    """
    # perform a translation registration and get parameters
    fixed_image, moving_image = resample_images_to_fit(fixed_image, moving_image)
    translated_image, parameter_map = translate_image(fixed_image, moving_image)
    transform_parameters = [float(x) for x in parameter_map["TransformParameters"]]

    # compute the final volume dimensions
    final_vol_dim = [
        x + abs(round(y / s))
        for x, y, s in zip(fixed_image.GetSize(), transform_parameters, fixed_image.GetSpacing())
    ]
    z_direction = (
        "caudal" if transform_parameters[2] < 0 else "cranial"
    )  # caudal -> the moving image is placed caudal the fixed image

    # compute volume affine data
    spacing = fixed_image.GetSpacing()
    origin = fixed_image.GetOrigin() if z_direction == "caudal" else moving_image.GetOrigin()
    direction = fixed_image.GetDirection()

    # create an empty volume with these dimensions
    empty_array = np.zeros(final_vol_dim[::-1])

    # fill the volume according to the transform paramters
    stitched_array = combine_images(empty_array, fixed_image, translated_image, z_direction)

    # convert array to sitk.Image and update affine
    stiched_image = sitk.GetImageFromArray(stitched_array)
    stiched_image.SetSpacing(spacing)
    stiched_image.SetOrigin(origin)
    stiched_image.SetDirection(direction)

    return stiched_image


def resample_to_isotropic(
    img: sitk.Image,
    new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator: int = sitk.sitkLinear,
) -> sitk.Image:
    """
    This function resamples an image to isotropic pixel spacing.

    Args:
        img: input SimpleITK image.
        new_spacing: desired pixel spacing.
        interpolator: interpolation method (default is sitk.sitkLinear).

    Returns:
        Resampled SimpleITK image.
    """
    # Original image spacing and size
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    # Compute new image size
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    # Resample image
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(img.GetPixelIDValue())
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(img)


def stitch_images(
    images: List[sitk.Image], new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> List[sitk.Image]:
    """
    Combine a list of images into a single large image with isotropnic spacing
    """
    isotrophic_images = [
        resample_to_isotropic(img, new_spacing=new_spacing)
        for img in tqdm(images, postfix="Resampling Series")
    ]
    fixed_image, *moving_images = isotrophic_images
    for moving in tqdm(moving_images, postfix="Stitching Series"):
        fixed_image = stitch_two_images(fixed_image, moving)
    return fixed_image
