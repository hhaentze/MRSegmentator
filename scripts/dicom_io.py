"""
Functions for reading and writing medical images in various formats
"""

import tempfile
import zipfile
from multiprocessing import Pool
from os.path import join
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm.autonotebook import tqdm


def read_dicom_series(
    directory: Union[str, Path], pbar_position=0
) -> (List[sitk.Image], List[str]):
    """
    This function reads one or multiple DICOM series from a directory and returns
        the resulting 3D image/s.

    Args:
        directory (str, Path): The path to the directory containing the DICOM series.

    Returns:
        sitk.Image: A list containing the resulting 3D image as SimpleITK Image objects.

    """
    directory = str(directory)
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory)
    try:
        manifest = pd.read_csv(join(directory, "manifest.cvs"))
    except FileNotFoundError:
        manifest = pd.read_csv(join(directory, "manifest.csv"))
    series_desc = [
        manifest.loc[manifest["seriesid"] == sid]["series discription"].iloc[0]
        for sid in series_ids
    ]
    images = []
    for idx in tqdm(
        series_ids, postfix="Reading multiple DICOM Series", leave=False, position=pbar_position
    ):
        file_names = reader.GetGDCMSeriesFileNames(directory, idx)
        reader.SetFileNames(file_names)
        image = reader.Execute()
        images.append(image)
    return images, series_desc


def read_dicom_series_zipped(
    file: Union[str, Path], pbar_position=0
) -> (List[sitk.Image], List[str]):
    """
    Reads One or multiple DICOM series from a zip file.

    Args:
        file: Path to the zip file containing the DICOM series.

    Returns:
        List of SimpleITK images, one for each DICOM series in the zip file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(file, "r") as zip_file:
            for name in zip_file.namelist():
                *parents, fn = name.split("/")
                zip_file.extract(name, temp_dir)
        return read_dicom_series(Path(temp_dir).joinpath(*parents), pbar_position)


def read_dicom_series_zipped_parallel(zip_file_paths: List[str], n_cpu: int):
    """
    Converts all DICOM series in a list of zip files to compressed NRRD volumes.

    :param zip_file_paths: List of paths to the zip files containing the DICOM files.
    """
    images = []
    with Pool(processes=n_cpu) as pool:
        for im in tqdm(
            pool.imap_unordered(read_dicom_series_zipped, zip_file_paths), total=len(zip_file_paths)
        ):
            images.append(im)
    return images


def extract_axial_sequence(series: List[sitk.Image]) -> sitk.Image:
    """
    Extract the axial sequence form a series of images

    :param series: List of sitk.Images of which one is probably an axial sequence
    """
    expected_direction = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)
    expected_minimal_slices = 30
    for image in series:
        *_, slices = image.GetSize()

        if (
            slices >= expected_minimal_slices
            and np.core.numeric.isclose(expected_direction, image.GetDirection(), rtol=0.1).all()
        ):
            array = sitk.GetArrayFromImage(image)
            # coronar = array[21:, :, :]
            axial = array[:21, :, :]
            axial = sitk.GetImageFromArray(axial)
            axial.SetSpacing(image.GetSpacing())
            axial.SetOrigin(image.GetOrigin())
            # we keep the direction as identity matrix
            return axial
