# Copyright 2024-2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""
Integration tests — requires model weights and real images.

Setup:
    1. cp tests/integration_config.example.py tests/integration_config.py
    2. Fill in MODEL_PATH and TEST_IMAGES
    3. pip install -e ".[dev]"

Run:
    make smoke   # fast, no model required
    make full    # smoke + integration
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Config — skip module if integration_config.py is absent or paths missing
# ---------------------------------------------------------------------------

try:
    from tests.mrsegmentator import integration_config as cfg
except ImportError:
    try:
        import integration_config as cfg
    except ImportError:
        cfg = None

if cfg is None:
    pytest.skip(
        "integration_config.py not found — copy integration_config.example.py and fill in paths.",
        allow_module_level=True,
    )

TEST_IMAGES: List[Path] = cfg.TEST_IMAGES

missing = [p for p in TEST_IMAGES if not p.exists()]
if missing:
    pytest.skip(f"TEST_IMAGES not found: {missing}", allow_module_level=True)

MAIN_MODEL_MAX_LABEL = 40
BODY_COMP_MAX_LABEL = 10
FIGURES_DIR = Path("reports/figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_labels(path: Path):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return set(int(v) for v in np.unique(arr))


def _spacing(path: Path):
    return sitk.ReadImage(str(path)).GetSpacing()


def _run_infer(images: List[Path], model_name: str, tmp_path_factory) -> List[Tuple[Path, Path]]:
    """
    Run infer() once for all images and return [(input, output), ...].
    """
    from mrsegmentator.inference import infer
    from mrsegmentator.utils import add_postfix

    outdir = tmp_path_factory.mktemp("seg")

    t0 = time.monotonic()
    try:
        infer([str(i) for i in images], outdir=str(outdir), fast=True, model_name=model_name)
    finally:
        elapsed = time.monotonic() - t0

    return [(img, outdir / add_postfix(img.name, "seg")) for img in images]


def _save_figure(ct_path: Path, mask_path: Path, stem: str) -> None:
    """Generate and save a slice figure to reports/figures/<stem>.png."""
    from tests.mrsegmentator.report_utils import save_slice_figure

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIGURES_DIR / f"{stem}.png"
    save_slice_figure(str(ct_path), str(mask_path), stem, str(out_png))
    print(f"[report] saved {out_png}", file=sys.__stderr__, flush=True)


# ---------------------------------------------------------------------------
# Session fixtures — each model runs inference exactly once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", params=["base", "body_comp"])
def segmentations(request, tmp_path_factory):
    """
    Runs inference and saves figures for each configured model exactly once.
    Automatically parameterizes any test that requests this fixture.
    """
    model_name = request.param
    pairs = _run_infer(TEST_IMAGES, model_name, tmp_path_factory)

    for img, mask in pairs:
        _save_figure(img, mask, f"{img.stem}_{model_name}")

    return pairs


# ---------------------------------------------------------------------------
# Main model tests
# ---------------------------------------------------------------------------


class TestModel:
    def test_all_outputs_exist(self, segmentations):
        for _, out in segmentations:
            assert out.exists(), f"Output file not created: {out}"

    def test_labels_in_valid_range(self, segmentations, request):

        if "body_comp" in request.node.name:
            max_label = BODY_COMP_MAX_LABEL
        else:
            max_label = MAIN_MODEL_MAX_LABEL

        for _, out in segmentations:
            unexpected = _read_labels(out) - set(range(max_label + 1))
            assert not unexpected, f"{out.name}: unexpected label(s) {unexpected}"

    def test_geometry_matches_input(self, segmentations):
        for img, out in segmentations:
            for got, want in zip(_spacing(out), _spacing(img)):
                assert (
                    abs(got - want) < 0.01
                ), f"{out.name}: spacing mismatch — got {_spacing(out)}, want {_spacing(img)}"

    def test_multiple_structures_present(self, segmentations):
        for _, out in segmentations:
            foreground = _read_labels(out) - {0}
            assert len(foreground) >= 5, (
                f"{out.name}: expected ≥5 foreground structures, "
                f"got {len(foreground)}: {foreground}"
            )

    def test_output_readable_by_sitkio(self, segmentations):
        from mrsegmentator.simpleitk_reader_writer import SimpleITKIO

        for _, out in segmentations:
            arr, props = SimpleITKIO().read_image(str(out))
            assert arr.ndim == 4, f"{out.name}: expected 4-D array"
            assert "spacing" in props
            assert "sitk_stuff" in props
