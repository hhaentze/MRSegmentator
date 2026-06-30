# Copyright 2024-2026 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import hashlib
import json
import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry – add new models here
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "base": {
        "version": 1.2,
        "url": "https://github.com/hhaentze/MRSegmentator/releases/download/v1.2.0/weights.zip",
        "sha256": "dd26c1b511908a1f2a752b7fe8f5ca9c5603fc3e69fa52f57bf3ccdd52ceb286",
        "zip_name": "mrsegmentator_weights.zip",
    },
    "body_comp": {
        "version": 1.0,
        "url": None,  # TODO: fill when secondary weights are published
        "sha256": "3ec490f641dd1aebdd4d2f497e69f8c9f2e4060a45c7d0d69c3ea0e74daf1550",
        "zip_name": "mrsegmentator_secondary_weights.zip",
    },
}

# Internal state – use is_legacy_mode() to read from outside the module.
_legacy_mode: bool = False


def is_legacy_mode() -> bool:
    """Whether MRSEG_WEIGHTS_PATH points at a raw nnUNet directory.

    Callers (e.g. argparse logic) can check this after setup_mrseg()
    to decide whether multi-model features are available.
    """
    return _legacy_mode


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------
def _resolve_root() -> Path:
    """Return the top-level weights root directory and detect legacy mode.

    Resolution order:
        1. MRSEG_WEIGHTS_PATH  +  plans.json at root  →  legacy mode
           (directory is treated as raw nnUNet weights; read-only,
            single-model, no downloads)
        2. MRSEG_WEIGHTS_PATH  without plans.json     →  custom directory
           (behaves identically to ~/.mrsegmentator/)
        3. ~/.mrsegmentator/                           →  default
    """
    global _legacy_mode

    if "MRSEG_WEIGHTS_PATH" in os.environ:
        root = Path(os.environ["MRSEG_WEIGHTS_PATH"])
        if not root.exists():
            raise FileNotFoundError(f"Could not find custom weights path {root}.")

        if (root / "plans.json").is_file():
            _legacy_mode = True
            logger.info(
                "Legacy mode: %s contains plans.json, treating as direct "
                "model weights. Multi-model features are disabled.",
                root,
            )
        else:
            _legacy_mode = False
            logger.debug("Using custom weights directory: %s", root)

        return root

    _legacy_mode = False
    root = Path.home() / ".mrsegmentator"
    root.mkdir(exist_ok=True)
    logger.debug("Using default weights directory: %s", root)
    return root


def _get_old_install_dir() -> Optional[Path]:
    """Return the old <module_dir>/weights/ path if it contains weights."""
    old_dir = Path(os.path.dirname(__file__)) / "weights"
    if old_dir.is_dir() and any(old_dir.iterdir()):
        return old_dir
    return None


def get_model_dir(model_name: str) -> Path:
    """Return the directory for a specific model's weights.

    In legacy mode the root directory itself is returned (it *is* the
    model).  Otherwise each model lives in its own subdirectory, with a
    silent fallback to the old <module_dir>/weights/ location for the
    base model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    root = _resolve_root()

    # Legacy mode: the root directory IS the model weights
    if _legacy_mode:
        if model_name != "base":
            raise ValueError(
                f"Model '{model_name}' is not available in legacy mode. "
                f"MRSEG_WEIGHTS_PATH points directly at model weights "
                f"(plans.json detected). To use multiple models, either "
                f"unset MRSEG_WEIGHTS_PATH or point it at a directory "
                f"containing base/ and secondary/ subdirectories."
            )
        return root

    # New-style location: <root>/<model_name>/
    model_dir = root / model_name
    if model_dir.is_dir() and any(model_dir.iterdir()):
        return model_dir

    # Old-install fallback (base model only, default path only)
    if model_name == "base" and "MRSEG_WEIGHTS_PATH" not in os.environ:
        old_dir = _get_old_install_dir()
        if old_dir is not None:
            logger.debug("Using base weights from old location %s.", old_dir)
            return old_dir

    return model_dir  # may not exist yet – caller handles download


# ---------------------------------------------------------------------------
# Version tracking (reads version.json from inside each model directory)
# ---------------------------------------------------------------------------
def _read_model_version(model_dir: Path) -> float:
    """Read the weights version from a model directory's version.json.

    Returns 0.0 if the directory or file does not exist.
    """
    version_file = model_dir / "version.json"
    if version_file.is_file():
        with open(version_file, "r") as f:
            config = json.load(f)
        return config.get("weights_version", 0.0)
    return 0.0


# ---------------------------------------------------------------------------
# Download & verification
# ---------------------------------------------------------------------------
def _verify_checksum(filepath: Path, expected_sha256: Optional[str]) -> bool:
    """Verify SHA-256 checksum. Returns True if no checksum is set (opt-in)."""
    if expected_sha256 is None:
        logger.debug("No checksum configured, skipping verification.")
        return True

    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected_sha256:
        logger.error(
            "Checksum mismatch for %s: expected %s, got %s",
            filepath,
            expected_sha256,
            actual,
        )
        return False
    logger.debug("Checksum verified for %s", filepath)
    return True


def _download_model(model_name: str) -> None:
    """Download, verify, extract, and clean up weights for a single model.

    The zip is expected to contain a version.json alongside the model files.
    After extraction that version.json serves as the version tracker.

    Skipped entirely in legacy mode (weights are user-managed).
    """
    if _legacy_mode:
        logger.debug("Legacy mode active, skipping download for '%s'.", model_name)
        return

    config = MODEL_REGISTRY[model_name]
    if config["url"] is None:
        raise RuntimeError(f"No download URL configured for model '{model_name}'.")

    root = _resolve_root()
    model_dir = root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    zip_path = model_dir / config["zip_name"]

    # Download
    logger.info("Downloading weights for '%s' to %s ...", model_name, model_dir)
    with urllib.request.urlopen(config["url"]) as response:
        file_size = int(response.info().get("Content-Length", -1))

    with tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=config["zip_name"],
    ) as pbar:

        def update_progress(block_num: int, block_size: int, total_size: int) -> None:
            if pbar.total != total_size:
                pbar.total = total_size
            pbar.update(block_num * block_size - pbar.n)

        urllib.request.urlretrieve(config["url"], zip_path, reporthook=update_progress)

    # Verify checksum
    if not _verify_checksum(zip_path, config["sha256"]):
        zip_path.unlink()
        raise RuntimeError(
            f"Checksum verification failed for '{model_name}'. "
            f"The corrupted download has been removed. Please try again."
        )

    # Extract
    logger.debug("Extracting weights for '%s'...", model_name)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(model_dir)

    # Verify extraction produced files, then clean up zip
    extracted_files = [p for p in model_dir.iterdir() if p != zip_path]
    if not extracted_files:
        raise RuntimeError(f"Zip extraction produced no files for '{model_name}'.")
    zip_path.unlink()

    installed_version = _read_model_version(model_dir)
    logger.info(
        "Weights for '%s' (v%s) installed at %s",
        model_name,
        installed_version,
        model_dir,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ensure_model(model_name: str) -> Path:
    """Ensure weights for the given model are available and up to date.

    In legacy mode no version check or download is performed – the
    directory is used as-is.  Otherwise downloads if missing or outdated.

    Returns the path to the model directory.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    model_dir = get_model_dir(model_name)

    # Legacy mode: no version management, use directory as-is
    if _legacy_mode:
        return model_dir

    expected_version = MODEL_REGISTRY[model_name]["version"]
    current_version = _read_model_version(model_dir)

    needs_download = (
        current_version < expected_version or not model_dir.is_dir() or not any(model_dir.iterdir())
    )

    if needs_download:
        if current_version > 0:
            logger.info(
                "Updating '%s' weights: v%s -> v%s",
                model_name,
                current_version,
                expected_version,
            )
        _download_model(model_name)
        # Re-resolve in case download just created the directory
        model_dir = get_model_dir(model_name)

    return model_dir


def disable_nnunet_path_warnings() -> None:
    """Disable warning message about undefined environmental variables.
    (We assign temporary arbitrary values. The script does not use these.)
    """
    for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if os.environ.get(var) is None:
            os.environ[var] = "empty"


def setup_mrseg(model_name: str = "base") -> Path:
    """Set up weights for the given model and return the model directory.

    This is the main entry point.  Call with no arguments for backward
    compatibility (base model), or with model_name="body_comp" for the
    secondary segmentation model.

    After this call, ``is_legacy_mode()`` indicates
    whether MRSEG_WEIGHTS_PATH pointed at a raw nnUNet directory.
    Callers should check this flag to decide whether multi-model
    features are available.
    """
    model_dir = ensure_model(model_name)

    version = _read_model_version(model_dir)
    logger.debug("Using '%s' model v%s for inference.", model_name, version)

    disable_nnunet_path_warnings()

    return model_dir
