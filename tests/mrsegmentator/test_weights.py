import json
import os

import pytest


class TestGetModelDir:
    """Tests for get_model_dir – the single public path function."""

    def test_default_returns_home_subdir(self, monkeypatch):
        from mrsegmentator.config import get_model_dir

        monkeypatch.delenv("MRSEG_WEIGHTS_PATH", raising=False)
        model_dir = get_model_dir("base")
        assert model_dir.name == "base"
        assert model_dir.parent.name == ".mrsegmentator"

    def test_default_body_comp_returns_home_subdir(self, monkeypatch):
        from mrsegmentator.config import get_model_dir

        monkeypatch.delenv("MRSEG_WEIGHTS_PATH", raising=False)
        model_dir = get_model_dir("body_comp")
        assert model_dir.name == "body_comp"
        assert model_dir.parent.name == ".mrsegmentator"

    def test_custom_dir_returns_subdir(self, tmp_path, monkeypatch):
        from mrsegmentator.config import get_model_dir

        monkeypatch.setenv("MRSEG_WEIGHTS_PATH", str(tmp_path))
        assert get_model_dir("base") == tmp_path / "base"
        assert get_model_dir("body_comp") == tmp_path / "body_comp"

    def test_missing_custom_path_raises(self, tmp_path, monkeypatch):
        from mrsegmentator.config import get_model_dir

        monkeypatch.setenv("MRSEG_WEIGHTS_PATH", str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            get_model_dir("base")

    def test_unknown_model_raises(self, monkeypatch):
        from mrsegmentator.config import get_model_dir

        monkeypatch.delenv("MRSEG_WEIGHTS_PATH", raising=False)
        with pytest.raises(ValueError, match="nonexistent"):
            get_model_dir("nonexistent")


class TestLegacyMode:
    """Tests for legacy mode – MRSEG_WEIGHTS_PATH with plans.json at root."""

    def test_plans_json_triggers_legacy(self, tmp_path, monkeypatch):
        from mrsegmentator.config import get_model_dir, is_legacy_mode

        (tmp_path / "plans.json").write_text("{}")
        monkeypatch.setenv("MRSEG_WEIGHTS_PATH", str(tmp_path))

        get_model_dir("base")
        assert is_legacy_mode() is True

    def test_legacy_base_returns_root(self, tmp_path, monkeypatch):
        from mrsegmentator.config import get_model_dir

        (tmp_path / "plans.json").write_text("{}")
        monkeypatch.setenv("MRSEG_WEIGHTS_PATH", str(tmp_path))

        assert get_model_dir("base") == tmp_path

    def test_legacy_body_comp_raises(self, tmp_path, monkeypatch):
        from mrsegmentator.config import get_model_dir

        (tmp_path / "plans.json").write_text("{}")
        monkeypatch.setenv("MRSEG_WEIGHTS_PATH", str(tmp_path))

        with pytest.raises(ValueError, match="legacy mode"):
            get_model_dir("body_comp")

    def test_no_plans_json_means_not_legacy(self, tmp_path, monkeypatch):
        from mrsegmentator.config import get_model_dir, is_legacy_mode

        monkeypatch.setenv("MRSEG_WEIGHTS_PATH", str(tmp_path))

        get_model_dir("base")
        assert is_legacy_mode() is False

    def test_ensure_model_skips_download_in_legacy(self, tmp_path, monkeypatch):
        from mrsegmentator.config import ensure_model

        (tmp_path / "plans.json").write_text("{}")
        (tmp_path / "fold_0").mkdir()
        monkeypatch.setenv("MRSEG_WEIGHTS_PATH", str(tmp_path))

        # Should not attempt any download, just return the path
        model_dir = ensure_model("base")
        assert model_dir == tmp_path


class TestLoadOrder:
    """The new location <root>/base/ must take priority over <module_dir>/weights/."""

    def test_new_location_takes_priority(self, tmp_path, monkeypatch):
        from mrsegmentator import config
        from mrsegmentator.config import get_model_dir

        monkeypatch.delenv("MRSEG_WEIGHTS_PATH", raising=False)

        # Create new-style location
        new_base = tmp_path / ".mrsegmentator" / "base"
        new_base.mkdir(parents=True)
        (new_base / "fold_0").mkdir()

        # Create old-style location
        old_dir = tmp_path / "old_weights"
        old_dir.mkdir()
        (old_dir / "fold_0").mkdir()

        # Patch both resolution paths
        monkeypatch.setattr(
            config, "_resolve_root", lambda: _set_and_return(config, tmp_path / ".mrsegmentator")
        )
        monkeypatch.setattr(config, "_get_old_install_dir", lambda: old_dir)

        model_dir = get_model_dir("base")
        assert model_dir == new_base

    def test_old_install_used_when_new_missing(self, tmp_path, monkeypatch):
        from mrsegmentator import config
        from mrsegmentator.config import get_model_dir

        monkeypatch.delenv("MRSEG_WEIGHTS_PATH", raising=False)

        # Only old-style location exists
        old_dir = tmp_path / "old_weights"
        old_dir.mkdir()
        (old_dir / "fold_0").mkdir()

        root = tmp_path / ".mrsegmentator"
        root.mkdir(parents=True)

        monkeypatch.setattr(config, "_resolve_root", lambda: _set_and_return(config, root))
        monkeypatch.setattr(config, "_get_old_install_dir", lambda: old_dir)

        model_dir = get_model_dir("base")
        assert model_dir == old_dir


class TestVersionReading:
    """Tests for _read_model_version."""

    def test_reads_version(self, tmp_path):
        from mrsegmentator.config import _read_model_version

        (tmp_path / "version.json").write_text(json.dumps({"weights_version": 1.2}))
        assert _read_model_version(tmp_path) == 1.2

    def test_missing_file_returns_zero(self, tmp_path):
        from mrsegmentator.config import _read_model_version

        assert _read_model_version(tmp_path) == 0.0

    def test_missing_key_returns_zero(self, tmp_path):
        from mrsegmentator.config import _read_model_version

        (tmp_path / "version.json").write_text(json.dumps({"other": 42}))
        assert _read_model_version(tmp_path) == 0.0

    def test_nonexistent_dir_returns_zero(self, tmp_path):
        from mrsegmentator.config import _read_model_version

        assert _read_model_version(tmp_path / "nonexistent") == 0.0


class TestSetupMrseg:
    """Tests for the main entry point."""

    def test_default_model_is_base(self):
        from mrsegmentator.config import setup_mrseg

        assert setup_mrseg.__defaults__ == ("base",)

    def test_disables_nnunet_warnings(self, monkeypatch):
        from mrsegmentator.config import disable_nnunet_path_warnings

        for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            monkeypatch.delenv(var, raising=False)

        disable_nnunet_path_warnings()

        for var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            assert var in os.environ

    def test_nnunet_does_not_overwrite_existing(self, monkeypatch):
        from mrsegmentator.config import disable_nnunet_path_warnings

        monkeypatch.setenv("nnUNet_raw", "/real/path")
        disable_nnunet_path_warnings()
        assert os.environ["nnUNet_raw"] == "/real/path"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _set_and_return(config_module, root):
    """Helper to patch _resolve_root: sets _legacy_mode = False and returns root."""
    config_module._legacy_mode = False
    return root
