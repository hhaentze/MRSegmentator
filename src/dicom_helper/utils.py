# Copyright 2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import os
from abc import ABC

import numpy as np
import pydicom
import SimpleITK as sitk
from highdicom.color import CIELabColor

logger = logging.getLogger(__name__)


def validate_dicom_directory(dicom_dir):
    """Validate that directory exists and contains DICOM files."""
    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"Directory not found: {dicom_dir}")

    # Check for at least one DICOM file
    has_dicom = False
    for file in os.listdir(dicom_dir):
        try:
            filepath = os.path.join(dicom_dir, file)
            if os.path.isfile(filepath):
                pydicom.dcmread(filepath, stop_before_pixels=True)
                has_dicom = True
                break
        except Exception as e:
            logger.warning(e)
            continue

    if not has_dicom:
        raise ValueError(f"No valid DICOM files found in {dicom_dir}")


def validate_nifti_file(nifti_path):
    """Validate that NIfTI file exists and is readable."""
    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    try:
        img = sitk.ReadImage(nifti_path)
        return img
    except Exception as e:
        raise ValueError(f"Cannot read NIfTI file {nifti_path}: {e}")


def validate_dtype(dtype_str):
    """Validate and return numpy dtype."""
    try:
        dtype = getattr(np, dtype_str)
        if not np.issubdtype(dtype, np.integer):
            raise ValueError(f"Only integer dtypes supported, got {dtype_str}")
        return dtype
    except AttributeError:
        raise ValueError(f"Invalid dtype: {dtype_str}")


def verify_geometry_match(seg_img, ref_img, tolerance=1e-3):
    """Verify that segmentation and reference images have compatible geometry."""
    if seg_img.GetSize() != ref_img.GetSize():
        raise ValueError(f"Size mismatch: seg={seg_img.GetSize()}, ref={ref_img.GetSize()}")

    seg_spacing = np.array(seg_img.GetSpacing())
    ref_spacing = np.array(ref_img.GetSpacing())
    if not np.allclose(seg_spacing, ref_spacing, rtol=tolerance):
        logger.warning(f"Spacing mismatch: seg={seg_spacing}, ref={ref_spacing}")

    seg_origin = np.array(seg_img.GetOrigin())
    ref_origin = np.array(ref_img.GetOrigin())
    if not np.allclose(seg_origin, ref_origin, atol=tolerance):
        logger.warning(f"Origin mismatch: seg={seg_origin}, ref={ref_origin}")


def is_dicom(file_path):
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        """Checks if a path points to a DICOM file."""
        return True
    except FileNotFoundError as e:
        raise e
    except Exception:
        return False


def has_dicom_file(directory_path):
    """Checks if a path is a directory and contains at least one DICOM file."""
    if not os.path.isdir(directory_path):
        return False

    for root, _, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if is_dicom(file_path):
                return True

    return False


class Color(ABC):
    """Cielab Color Helper"""

    @staticmethod
    def rgb_to_xyz(r, g, b):
        """Convert sRGB to CIE 1931 XYZ (D65 white point)"""
        # Normalize RGB to [0, 1]
        r, g, b = [x / 255.0 for x in (r, g, b)]

        # Apply gamma correction (inverse sRGB)
        def inverse_gamma(c):
            return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92

        r = inverse_gamma(r)
        g = inverse_gamma(g)
        b = inverse_gamma(b)

        # Convert to XYZ using sRGB matrix
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        return x, y, z

    @staticmethod
    def xyz_to_lab(x, y, z):
        """Convert XYZ to CIELab using D65 reference white"""
        # Reference white point (D65)
        xr, yr, zr = 0.95047, 1.00000, 1.08883

        def f(t):
            return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

        fx, fy, fz = f(x / xr), f(y / yr), f(z / zr)

        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return L, a, b

    @staticmethod
    def rgb_to_cielab(rgb):
        r, g, b = rgb
        x, y, z = Color.rgb_to_xyz(r, g, b)
        L, a, b = Color.xyz_to_lab(x, y, z)

        return CIELabColor(L, a, b)
