# Copyright 2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import os
import tempfile

import highdicom as hd
import numpy as np
import pydicom
import SimpleITK as sitk
from highdicom.seg.sop import Segmentation
from pydicom.uid import generate_uid

from dicom_helper import utils
from dicom_helper.class_codes import CLASS_CODES
from dicom_helper.class_colors import CLASS_COLORS

logger = logging.getLogger(__name__)


def list_series(dicom_dir):
    """List available DICOM series in a directory."""
    utils.validate_dicom_directory(dicom_dir)

    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not ids:
        logger.warning(f"No DICOM series found in {dicom_dir}")
        return []

    info = []
    for series in ids:
        try:
            files = reader.GetGDCMSeriesFileNames(dicom_dir, series)
            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            desc = getattr(ds, "SeriesDescription", "Unknown")
            modality = getattr(ds, "Modality", "Unknown")
            info.append((series, desc, modality, len(files)))
        except Exception as e:
            logger.warning(f"Error reading series {series}: {e}")
            continue

    return info


def dicom_to_nifti(dicom_dir, output_dir=None, series_id=None, output_name="image.nii.gz"):
    """Convert a DICOM series to NIfTI file."""
    utils.validate_dicom_directory(dicom_dir)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="d2n_")
    os.makedirs(output_dir, exist_ok=True)

    reader = sitk.ImageSeriesReader()
    all_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not all_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")

    if series_id is None:
        if len(all_ids) > 1:
            series_info = list_series(dicom_dir)
            logger.warning("Multiple series found; using the first one.\nAvailable series:")
            for i, (sid, desc, mod, count) in enumerate(series_info):
                logger.info(f"  {i}: {sid} - {desc} ({mod}, {count} files)")
        chosen = all_ids[0]
    else:
        if series_id not in all_ids:
            raise ValueError(f"Series ID {series_id} not found. Available: {all_ids}")
        chosen = series_id

    filenames = reader.GetGDCMSeriesFileNames(dicom_dir, chosen)
    reader.SetFileNames(filenames)

    try:
        image = reader.Execute()
    except Exception as e:
        raise RuntimeError(f"Failed to read DICOM series: {e}")

    # Ensure consistent orientation - SimpleITK handles LPS internally
    output_file = os.path.join(output_dir, output_name)
    sitk.WriteImage(image, output_file)
    logger.info(f"Saved NIfTI to {output_file}")
    return output_file


def nifti_to_dicom_slices(nifti_path, template_dir, output_dir, dtype="uint8", multiclass=True):
    """Convert NIfTI segmentation back to DICOM slices."""
    # Validate inputs
    seg_img = utils.validate_nifti_file(nifti_path)
    utils.validate_dicom_directory(template_dir)
    dtype = utils.validate_dtype(dtype)
    os.makedirs(output_dir, exist_ok=True)

    # Read reference DICOM series
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(template_dir)
    if not ids:
        raise FileNotFoundError(f"No DICOM series in {template_dir}")

    files = reader.GetGDCMSeriesFileNames(template_dir, ids[0])
    reader.SetFileNames(files)
    ref_img = reader.Execute()

    # Copy spatial information from reference to segmentation
    seg_img.CopyInformation(ref_img)

    # Verify geometry compatibility
    utils.verify_geometry_match(seg_img, ref_img)

    # Convert to array and ensure proper orientation
    seg_array = sitk.GetArrayFromImage(seg_img)

    # Analyze labels
    unique_labels = np.unique(seg_array)
    foreground_labels = unique_labels[unique_labels > 0]

    logger.info(f"Found labels: {unique_labels}")

    if len(foreground_labels) == 0:
        raise ValueError("No foreground labels found in segmentation")

    # Handle multiclass segmentations
    if multiclass and len(foreground_labels) > 1:
        logger.info(f"Creating multiclass DICOM slices for {len(foreground_labels)} labels")
        # Keep original labels for multiclass
        output_array = seg_array.astype(dtype)
    else:
        if len(foreground_labels) > 1:
            logger.info(
                f"Converting multiclass segmentation to binary (combining labels {foreground_labels})"
            )
        # Convert to binary
        output_array = (seg_array > 0).astype(dtype)

    if output_array.shape[0] != len(files):
        raise ValueError(
            f"Slice count mismatch: segmentation has {output_array.shape[0]} slices, "
            f"template has {len(files)} files"
        )

    # Create new series UID
    new_series_uid = generate_uid()

    # Determine series description
    if multiclass and len(foreground_labels) > 1:
        series_desc = f"AI Segmentation ({len(foreground_labels)} classes)"
    else:
        series_desc = "AI Segmentation (binary)"

    # Process each slice
    saved_count = 0
    for idx in range(output_array.shape[0]):
        slice_data = output_array[idx, ...]

        try:
            # Read original DICOM
            original_ds = pydicom.dcmread(files[idx])

            # Create new dataset for segmentation
            seg_ds = pydicom.Dataset()

            # Copy essential metadata
            seg_ds.PatientName = getattr(original_ds, "PatientName", "")
            seg_ds.PatientID = getattr(original_ds, "PatientID", "")
            seg_ds.StudyInstanceUID = original_ds.StudyInstanceUID
            seg_ds.StudyID = getattr(original_ds, "StudyID", "")
            seg_ds.StudyDate = getattr(original_ds, "StudyDate", "")
            seg_ds.StudyTime = getattr(original_ds, "StudyTime", "")

            # Set segmentation-specific metadata
            seg_ds.Modality = "SEG"
            seg_ds.SeriesInstanceUID = new_series_uid
            seg_ds.SeriesNumber = getattr(original_ds, "SeriesNumber", 1) + 1000
            seg_ds.SeriesDescription = series_desc

            # Copy spatial metadata
            seg_ds.ImagePositionPatient = original_ds.ImagePositionPatient
            seg_ds.ImageOrientationPatient = original_ds.ImageOrientationPatient
            seg_ds.PixelSpacing = original_ds.PixelSpacing
            seg_ds.SliceThickness = getattr(original_ds, "SliceThickness", "")

            # Set image data
            seg_ds.Rows, seg_ds.Columns = slice_data.shape
            seg_ds.PixelData = slice_data.tobytes()

            # Set pixel data characteristics
            seg_ds.SamplesPerPixel = 1
            seg_ds.PhotometricInterpretation = "MONOCHROME2"
            seg_ds.BitsAllocated = slice_data.dtype.itemsize * 8
            seg_ds.BitsStored = seg_ds.BitsAllocated
            seg_ds.HighBit = seg_ds.BitsStored - 1
            seg_ds.PixelRepresentation = 0

            # Generate unique IDs
            seg_ds.SOPInstanceUID = generate_uid()
            seg_ds.InstanceNumber = idx + 1

            # Set required DICOM metadata
            seg_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage
            seg_ds.TransferSyntaxUID = "1.2.840.10008.1.2.1"  # Explicit VR Little Endian

            # Save file
            output_file = os.path.join(output_dir, f"seg_{idx+1:03d}.dcm")
            seg_ds.save_as(output_file)
            saved_count += 1

        except Exception as e:
            logger.error(f"Error processing slice {idx}: {e}")
            continue

    logger.info(f"Saved {saved_count} DICOM segmentation slices to {output_dir}")
    if multiclass and len(foreground_labels) > 1:
        logger.info(f"Multiclass segmentation preserved with labels: {foreground_labels}")


def nifti_to_dicom_seg(nifti_path, template_dir, output_file, multiclass=True):
    """Create a DICOM SEG object from a NIfTI segmentation using highdicom."""

    # Validate inputs
    seg_img = utils.validate_nifti_file(nifti_path)
    utils.validate_dicom_directory(template_dir)

    # Read reference DICOM series
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(template_dir)
    files = reader.GetGDCMSeriesFileNames(template_dir, ids[0])
    reader.SetFileNames(files)
    ref_img = reader.Execute()

    # Copy spatial information and verify geometry
    seg_img.CopyInformation(ref_img)
    utils.verify_geometry_match(seg_img, ref_img)

    # Get segmentation array and analyze labels
    # pixel_array = sitk.GetArrayFromImage(seg_img) # TODO remove the uint
    pixel_array = sitk.GetArrayViewFromImage(seg_img).astype(np.uint8)
    # pixel_array = np.flip(pixel_array, axis=1)  # TODO fix dicom/nifti orientation problems

    unique_labels = np.unique(pixel_array)
    unique_labels = unique_labels[unique_labels > 0]  # Remove background (0)

    logger.info(f"Found labels: {unique_labels}")

    if len(unique_labels) == 0:
        raise ValueError("No foreground labels found in segmentation")

    # Prepare pixel data based on multiclass setting
    if multiclass and len(unique_labels) > 1:
        # Create multi-label segmentation
        logger.info(f"Creating multiclass DICOM SEG with {len(unique_labels)} segments")

        # Create separate binary masks for each label
        segment_arrays = []
        # for label in CLASS_CODES:
        for label in unique_labels:
            binary_mask = (pixel_array == label).astype(np.uint8)
            segment_arrays.append(binary_mask)

        # Stack arrays - shape should be (segments, slices, height, width)
        pixel_data = np.stack(segment_arrays, axis=3)
        logger.info(f"Pixel data shape for SEG: {pixel_data.shape}")

    else:
        # Create binary segmentation (all foreground labels combined)
        if len(unique_labels) > 1:
            logger.info(
                f"Converting multiclass segmentation to binary (combining labels {unique_labels})"
            )
        binary_array = (pixel_array > 0).astype(np.uint8)
        pixel_data = binary_array[np.newaxis, ...]  # Add segment dimension

    # Read reference DICOM headers
    refs = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            refs.append(ds)
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")
            raise

    manufacturer = getattr(refs[0], "Manufacturer", "Unknown")
    manufacturer_model_name = getattr(refs[0], "ManufacturerModelName", "Unknown")
    software_versions = getattr(refs[0], "SoftwareVersions", "1.0")
    device_serial_number = getattr(refs[0], "DeviceSerialNumber", "0000")

    # Create segment descriptions
    segment_descriptions = []

    # Create description for each segment
    for i, label in enumerate(unique_labels, start=1):
        segment_descriptions.append(
            hd.seg.SegmentDescription(
                segment_number=i,
                segment_label=CLASS_CODES[label].meaning,
                segmented_property_category=hd.sr.CodedConcept(
                    value="123037004", scheme_designator="SCT", meaning="Anatomical Structure"
                ),
                segmented_property_type=CLASS_CODES[label],
                algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=hd.AlgorithmIdentificationSequence(
                    name="MRSegmentator",
                    version="1.2.3",
                    family=hd.sr.CodedConcept(
                        value="113011", scheme_designator="DCM", meaning="Algorithm"
                    ),
                ),
                display_color=utils.Color.rgb_to_cielab(CLASS_COLORS[label]),
            )
        )

    try:
        # Create segmentation
        logger.debug(f"Pixel data shape: {pixel_data.shape}")
        logger.debug(f"Unique labels: {unique_labels}")
        logger.debug(f"Num segment_descriptions: {len(segment_descriptions)}")
        assert pixel_data.shape[-1] == len(segment_descriptions), "Mismatch in segment count"

        segment = Segmentation(
            source_images=refs,
            pixel_array=pixel_data,
            segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=segment_descriptions,
            series_description="MRSegmentator Segmentation",
            series_instance_uid=generate_uid(),
            series_number=400,
            sop_instance_uid=generate_uid(),
            instance_number=1,
            manufacturer=manufacturer,
            manufacturer_model_name=manufacturer_model_name,
            software_versions=software_versions,
            device_serial_number=device_serial_number,
        )

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        segment.save_as(output_file)
        logger.info(f"Saved DICOM SEG with {len(segment_descriptions)} segment(s) to {output_file}")

    except Exception as e:
        raise RuntimeError(f"Failed to create DICOM SEG: {e}")
