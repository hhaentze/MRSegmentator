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

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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

MODEL_PATH: Path = cfg.MODEL_PATH
TEST_IMAGES: List[Path] = cfg.TEST_IMAGES
BODY_COMP_MODEL_PATH: Optional[Path] = getattr(cfg, "BODY_COMP_MODEL_PATH", None)

if not MODEL_PATH.exists():
    pytest.skip(f"MODEL_PATH does not exist: {MODEL_PATH}", allow_module_level=True)

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


def _run_infer(images: List[Path], weights: Path, tmp_path_factory) -> List[Tuple[Path, Path]]:
    """
    Run infer() once for all images and return [(input, output), ...].
    """
    from mrsegmentator.inference import infer
    from mrsegmentator.utils import add_postfix

    outdir = tmp_path_factory.mktemp("seg")

    old_env = os.environ.get("MRSEG_WEIGHTS_PATH")
    os.environ["MRSEG_WEIGHTS_PATH"] = str(weights)

    try:
        infer([str(i) for i in images], outdir=str(outdir), folds=[0], fast=True)
    finally:
        if old_env is None:
            os.environ.pop("MRSEG_WEIGHTS_PATH", None)
        else:
            os.environ["MRSEG_WEIGHTS_PATH"] = old_env

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


@pytest.fixture(scope="session")
def main_segs(tmp_path_factory):
    """Run main-model inference on all TEST_IMAGES; return [(in, out), ...]."""
    pairs = _run_infer(TEST_IMAGES, MODEL_PATH, tmp_path_factory)
    for ct, mask in pairs:
        _save_figure(ct, mask, f"{ct.stem}_main")
    return pairs


@pytest.fixture(scope="session")
def body_comp_segs(tmp_path_factory):
    """Run body-comp model inference. Skipped if BODY_COMP_MODEL_PATH is not set."""
    if BODY_COMP_MODEL_PATH is None or not BODY_COMP_MODEL_PATH.exists():
        pytest.skip("BODY_COMP_MODEL_PATH not configured — skipping body-comp tests.")
    pairs = _run_infer(TEST_IMAGES, BODY_COMP_MODEL_PATH, tmp_path_factory)
    for ct, mask in pairs:
        _save_figure(ct, mask, f"{ct.stem}_body_comp")
    return pairs


# ---------------------------------------------------------------------------
# Main model tests
# ---------------------------------------------------------------------------


class TestModel:
    def test_all_outputs_exist(self, main_segs):
        for _, out in main_segs:
            assert out.exists(), f"Output file not created: {out}"

    def test_labels_in_valid_range(self, main_segs):
        for _, out in main_segs:
            unexpected = _read_labels(out) - set(range(MAIN_MODEL_MAX_LABEL + 1))
            assert not unexpected, f"{out.name}: unexpected label(s) {unexpected}"

    def test_geometry_matches_input(self, main_segs):
        for ct, out in main_segs:
            for got, want in zip(_spacing(out), _spacing(ct)):
                assert (
                    abs(got - want) < 0.01
                ), f"{out.name}: spacing mismatch — got {_spacing(out)}, want {_spacing(ct)}"

    def test_multiple_structures_present(self, main_segs):
        for _, out in main_segs:
            foreground = _read_labels(out) - {0}
            assert len(foreground) >= 5, (
                f"{out.name}: expected ≥5 foreground structures, "
                f"got {len(foreground)}: {foreground}"
            )

    def test_output_readable_by_sitkio(self, main_segs):
        from mrsegmentator.simpleitk_reader_writer import SimpleITKIO

        for _, out in main_segs:
            arr, props = SimpleITKIO().read_image(str(out))
            assert arr.ndim == 4, f"{out.name}: expected 4-D array"
            assert "spacing" in props
            assert "sitk_stuff" in props
