"""
Unit tests for individual image preprocessors.

Heavy optional dependencies (SimpleITK, antspyx) are guarded with
pytest.importorskip so the suite runs in environments that do not have
them installed.  Lightweight logic (factory lookup, config parsing) is
tested without any image I/O.
"""

from __future__ import annotations

import pytest

from habit.core.preprocessing.preprocessor_factory import PreprocessorFactory


# ---------------------------------------------------------------------------
# PreprocessorFactory
# ---------------------------------------------------------------------------


class TestPreprocessorFactory:
    """Verify that the factory returns the right classes for known step names."""

    KNOWN_STEPS = [
        "n4_correction",
        "histogram_standardization",
        "zscore_normalization",
        "resample",
        "registration",
        "load_image",
        "adaptive_histogram_equalization",
    ]

    @pytest.mark.parametrize("step_name", KNOWN_STEPS)
    def test_factory_returns_object_for_known_step(self, step_name: str) -> None:
        """Factory must not raise for any registered step name."""
        preprocessor = PreprocessorFactory.create(step_name, params={})
        assert preprocessor is not None

    def test_factory_raises_for_unknown_step(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            PreprocessorFactory.create("nonexistent_step_xyz", params={})


# ---------------------------------------------------------------------------
# N4 bias-field correction (requires SimpleITK)
# ---------------------------------------------------------------------------


class TestN4Correction:
    sitk = pytest.importorskip("SimpleITK", reason="SimpleITK not installed")

    def test_n4_correction_instantiation(self) -> None:
        from habit.core.preprocessing.n4_correction import N4BiasFieldCorrection

        step = N4BiasFieldCorrection(params={})
        assert step is not None

    def test_n4_correction_has_transform_method(self) -> None:
        from habit.core.preprocessing.n4_correction import N4BiasFieldCorrection

        step = N4BiasFieldCorrection(params={})
        assert callable(getattr(step, "transform", None)) or callable(
            getattr(step, "process", None)
        )


# ---------------------------------------------------------------------------
# Resample (requires SimpleITK)
# ---------------------------------------------------------------------------


class TestResample:
    sitk = pytest.importorskip("SimpleITK", reason="SimpleITK not installed")

    def test_resample_instantiation(self) -> None:
        from habit.core.preprocessing.resample import Resample

        step = Resample(params={"spacing": [1.0, 1.0, 1.0]})
        assert step is not None


# ---------------------------------------------------------------------------
# ZScore normalisation (numpy only, no heavy deps)
# ---------------------------------------------------------------------------


class TestZScoreNormalization:
    def test_zscore_instantiation(self) -> None:
        pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
        from habit.core.preprocessing.zscore_normalization import ZScoreNormalization

        step = ZScoreNormalization(params={})
        assert step is not None

    def test_zscore_is_base_preprocessor(self) -> None:
        pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
        from habit.core.preprocessing.base_preprocessor import BasePreprocessor
        from habit.core.preprocessing.zscore_normalization import ZScoreNormalization

        assert issubclass(ZScoreNormalization, BasePreprocessor)


# ---------------------------------------------------------------------------
# BasePreprocessor interface contract
# ---------------------------------------------------------------------------


class TestBasePreprocessorContract:
    def test_base_preprocessor_is_abstract(self) -> None:
        import inspect

        pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
        from habit.core.preprocessing.base_preprocessor import BasePreprocessor

        abstract_methods = getattr(BasePreprocessor, "__abstractmethods__", set())
        assert len(abstract_methods) > 0, "BasePreprocessor should declare abstract methods"
