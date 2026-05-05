"""
Unit tests for preprocessing configuration schemas (PreprocessingConfig).

Covers: required fields, default values, field validators, and rejection of
invalid inputs — without touching the filesystem or any image library.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from habit.core.preprocessing.config_schemas import (
    PreprocessingConfig,
    PreprocessingStepConfig,
    SaveOptionsConfig,
)


# ---------------------------------------------------------------------------
# SaveOptionsConfig
# ---------------------------------------------------------------------------


class TestSaveOptionsConfig:
    def test_defaults(self) -> None:
        cfg = SaveOptionsConfig()
        assert cfg.save_intermediate is False
        assert cfg.intermediate_steps == []

    def test_explicit_values(self) -> None:
        cfg = SaveOptionsConfig(save_intermediate=True, intermediate_steps=["n4", "reg"])
        assert cfg.save_intermediate is True
        assert cfg.intermediate_steps == ["n4", "reg"]


# ---------------------------------------------------------------------------
# PreprocessingStepConfig
# ---------------------------------------------------------------------------


class TestPreprocessingStepConfig:
    def test_valid_images(self) -> None:
        cfg = PreprocessingStepConfig(images=["T1w.nii.gz"])
        assert cfg.images == ["T1w.nii.gz"]

    def test_empty_images_rejected(self) -> None:
        with pytest.raises(ValidationError, match="images must not be empty"):
            PreprocessingStepConfig(images=[])

    def test_extra_fields_allowed(self) -> None:
        """extra='allow' means arbitrary step params should not raise."""
        cfg = PreprocessingStepConfig(images=["img.nii"], some_param=123)
        assert cfg.some_param == 123  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# PreprocessingConfig
# ---------------------------------------------------------------------------


class TestPreprocessingConfig:
    def _minimal(self) -> dict:
        return {"data_dir": "/data", "out_dir": "/out"}

    def test_minimal_valid(self) -> None:
        cfg = PreprocessingConfig(**self._minimal())
        assert cfg.data_dir == "/data"
        assert cfg.out_dir == "/out"
        assert cfg.processes == 1
        assert cfg.random_state == 42
        assert cfg.auto_select_first_file is True

    def test_missing_data_dir_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PreprocessingConfig(out_dir="/out")

    def test_missing_out_dir_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PreprocessingConfig(data_dir="/data")

    def test_empty_data_dir_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PreprocessingConfig(data_dir="", out_dir="/out")

    def test_empty_out_dir_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PreprocessingConfig(data_dir="/data", out_dir="")

    def test_processes_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="processes must be >= 1"):
            PreprocessingConfig(**self._minimal(), processes=0)

    def test_processes_default_one(self) -> None:
        cfg = PreprocessingConfig(**self._minimal())
        assert cfg.processes == 1

    def test_preprocessing_steps_parsed(self) -> None:
        cfg = PreprocessingConfig(
            **self._minimal(),
            Preprocessing={"n4": {"images": ["T1w.nii.gz"]}},
        )
        assert "n4" in cfg.Preprocessing
        assert cfg.Preprocessing["n4"].images == ["T1w.nii.gz"]

    def test_save_options_defaults(self) -> None:
        cfg = PreprocessingConfig(**self._minimal())
        assert cfg.save_options.save_intermediate is False

    def test_custom_random_state(self) -> None:
        cfg = PreprocessingConfig(**self._minimal(), random_state=7)
        assert cfg.random_state == 7
