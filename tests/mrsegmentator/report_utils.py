# tests/report_utils.py
#
# Generates a three-panel MRI + segmentation figure and saves it as a PNG.
# Called directly from integration tests.

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import SimpleITK as sitk

MAX_LABELS = 40


def _resample_to_isotropic(image: sitk.Image, interpolator) -> sitk.Image:
    """Resample image to 1mm isotropic spacing for undistorted coronal/sagittal views."""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_spacing = (1.0, 1.0, 1.0)
    new_size = [int(round(orig_size[i] * orig_spacing[i])) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def _normalize_mri(arr: np.ndarray) -> np.ndarray:
    """
    Percentile-based normalization for MRI.

    MRI intensities have no absolute scale (unlike CT HU values), so we clip
    to the 1st–99th percentile of the image and rescale to [0, 1].
    """
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip(arr, p1, p99)
    return (arr - p1) / max(float(p99 - p1), 1e-8)


def _label_colors() -> dict:
    """RGBA colors for labels 1–40, cycled through tab20."""
    cmap = plt.cm.get_cmap("tab20", 20)
    return {i: (*cmap((i - 1) % 20)[:3], 0.45) for i in range(1, MAX_LABELS + 1)}


def save_slice_figure(mri_path: str, mask_path: str, title: str, out_png: str) -> None:
    """
    Save a three-panel (axial / coronal / sagittal) figure to `out_png`.

    Both images are reoriented to LPS and resampled to 1 mm isotropic so all
    three planes have correct aspect ratios and a consistent radiological
    orientation:
      axial    — patient right on left, anterior at bottom
      coronal  — patient right on left, superior at top
      sagittal — anterior on left,      superior at top
    """
    mri = sitk.ReadImage(mri_path)
    mask = sitk.ReadImage(mask_path)

    # Resample MRI into mask space (handles any geometry mismatch)
    mri = sitk.Resample(mri, mask, sitk.Transform(), sitk.sitkLinear, 0.0, mri.GetPixelID())

    # Reorient to LPS so array axes map to (Z=superior, Y=posterior, X=left)
    mri = sitk.DICOMOrient(mri, "LPS")
    mask = sitk.DICOMOrient(mask, "LPS")

    mri = _resample_to_isotropic(mri, sitk.sitkLinear)
    mask = _resample_to_isotropic(mask, sitk.sitkNearestNeighbor)

    mri_arr = sitk.GetArrayFromImage(mri).astype(np.float32)  # shape (Z, Y, X)
    mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)

    mri_arr = _normalize_mri(mri_arr)

    fg = np.argwhere(mask_arr > 0)
    if len(fg) > 0:
        z, y, x = fg.mean(axis=0).astype(int)
    else:
        z, y, x = [s // 2 for s in mask_arr.shape]

    colors = _label_colors()
    labels_present = sorted({int(v) for v in np.unique(mask_arr)} - {0})

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    for ax, (mri_slice, mask_slice, panel_label) in zip(
        axes,
        [
            (mri_arr[z, :, :], mask_arr[z, :, :], f"Axial z={z}"),
            (mri_arr[:, y, :], mask_arr[:, y, :], f"Coronal y={y}"),
            (mri_arr[:, :, x], mask_arr[:, :, x], f"Sagittal x={x}"),
        ],
    ):
        ax.imshow(mri_slice, cmap="gray", origin="lower", interpolation="nearest")
        for lbl in labels_present:
            overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
            overlay[mask_slice == lbl] = colors[lbl]
            ax.imshow(overlay, origin="lower", interpolation="nearest")
        ax.set_title(panel_label, fontsize=9)
        ax.axis("off")

    if labels_present:
        patches = [mpatches.Patch(color=colors[l][:3], label=f"Label {l}") for l in labels_present]
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=min(len(patches), 10),
            fontsize=7,
            frameon=False,
            bbox_to_anchor=(0.5, -0.03),
        )

    plt.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
