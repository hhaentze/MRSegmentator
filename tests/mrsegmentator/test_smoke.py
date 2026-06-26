# Copyright 2024-2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""
Smoke tests — no model weights or real images required.  All tests use
synthetic data created in memory and run in a few seconds on CPU.

Run with:
    pytest tests/mrsegmentator/test_smoke.py -v
    make smoke
"""

import argparse
import os

import numpy as np
import pytest
import SimpleITK as sitk


class TestIsSupported:
    def test_valid_extensions(self):
        from mrsegmentator.utils import is_supported

        assert is_supported("image.nii")
        assert is_supported("image.nii.gz")
        assert is_supported("image.mha")
        assert is_supported("image.nrrd")

    def test_path_prefix_accepted(self):
        from mrsegmentator.utils import is_supported

        assert is_supported("/some/dir/image.mha")

    def test_invalid_extensions_rejected(self):
        from mrsegmentator.utils import is_supported

        assert not is_supported("image.dcm")
        assert not is_supported("image.jpg")
        assert not is_supported("image.txt")
        assert not is_supported("image")


class TestReadImages:
    def test_single_valid_file(self, tmp_path):
        from mrsegmentator.utils import read_images

        f = tmp_path / "scan.mha"
        f.touch()
        assert read_images(str(f)) == [str(f)]

    def test_directory_returns_only_supported_files(self, tmp_path):
        from mrsegmentator.utils import read_images

        (tmp_path / "a.mha").touch()
        (tmp_path / "b.nii.gz").touch()
        (tmp_path / "notes.txt").touch()
        result = read_images(str(tmp_path))
        assert len(result) == 2

    def test_empty_directory_raises(self, tmp_path):
        from mrsegmentator.utils import read_images

        with pytest.raises(FileNotFoundError):
            read_images(str(tmp_path))

    def test_unsupported_file_raises(self, tmp_path):
        from mrsegmentator.utils import read_images

        f = tmp_path / "scan.dcm"
        f.touch()
        with pytest.raises(ValueError):
            read_images(str(f))


class TestSplitAndStitch:
    def _volume(self, depth=20):
        return np.zeros((1, depth, 8, 8), dtype=np.float32)

    def test_split_produces_two_overlapping_halves(self):
        from mrsegmentator.utils import split_image

        img = self._volume(depth=20)
        img1, img2 = split_image(img, margin=3)
        assert img1.shape[1] + img2.shape[1] == img.shape[1] + 2 * 3

    def test_stitch_recovers_original_depth(self):
        from mrsegmentator.utils import split_image, stitch_segmentations

        depth, margin = 20, 3
        img = self._volume(depth=depth)
        img1, img2 = split_image(img, margin=margin)
        stitched = stitch_segmentations(img1[0], img2[0], margin=margin)
        assert stitched.shape[0] == depth

    def test_zero_margin_round_trips(self):
        from mrsegmentator.utils import split_image, stitch_segmentations

        img = self._volume(depth=20)
        img1, img2 = split_image(img, margin=0)
        stitched = stitch_segmentations(img1[0], img2[0], margin=0)
        assert stitched.shape[0] == img.shape[1]


class TestParserValidation:
    def _ns(self, tmp_path, **overrides):
        defaults = dict(
            input=str(tmp_path),
            outdir=str(tmp_path / "out"),
            batchsize=8,
            nproc=3,
            nproc_export=8,
            split_level=0,
            split_margin=3,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_valid_namespace_passes(self, tmp_path):
        from mrsegmentator.parser import assert_namespace

        assert_namespace(self._ns(tmp_path))

    def test_batchsize_zero_raises(self, tmp_path):
        from mrsegmentator.parser import assert_namespace

        with pytest.raises(AssertionError):
            assert_namespace(self._ns(tmp_path, batchsize=0))

    def test_negative_split_level_raises(self, tmp_path):
        from mrsegmentator.parser import assert_namespace

        with pytest.raises(AssertionError):
            assert_namespace(self._ns(tmp_path, split_level=-1))

    def test_nonexistent_input_raises(self, tmp_path):
        from mrsegmentator.parser import assert_namespace

        with pytest.raises(AssertionError):
            assert_namespace(self._ns(tmp_path, input=str(tmp_path / "ghost.nii.gz")))


class TestOrientationString:
    VALID_CHARS = set("RLAPSI")

    def _image(self, direction):
        img = sitk.Image(5, 5, 5, sitk.sitkUInt8)
        img.SetDirection(direction)
        return img

    def test_returns_three_valid_chars(self):
        from mrsegmentator.simpleitk_reader_writer import get_orientation_string

        result = get_orientation_string(self._image([1, 0, 0, 0, 1, 0, 0, 0, 1]))
        assert len(result) == 3
        assert all(c in self.VALID_CHARS for c in result)

    def test_identity_direction_is_lps(self):
        from mrsegmentator.simpleitk_reader_writer import get_orientation_string

        assert get_orientation_string(self._image([1, 0, 0, 0, 1, 0, 0, 0, 1])) == "LPS"

    def test_flipped_x_axis_starts_with_r(self):
        from mrsegmentator.simpleitk_reader_writer import get_orientation_string

        result = get_orientation_string(self._image([-1, 0, 0, 0, 1, 0, 0, 0, 1]))
        assert result[0] == "R"


class TestConfigWeightsDir:
    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        from mrsegmentator.config import get_weights_dir

        monkeypatch.setenv("MRSEG_WEIGHTS_PATH", str(tmp_path))
        assert get_weights_dir() == tmp_path

    def test_missing_custom_path_raises(self, tmp_path, monkeypatch):
        from mrsegmentator.config import get_weights_dir

        monkeypatch.setenv("MRSEG_WEIGHTS_PATH", str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            get_weights_dir()
