"""
Unit tests for individual image preprocessors.

Heavy optional dependencies (SimpleITK, antspyx) are guarded with
pytest.importorskip so the suite runs in environments that do not have
them installed.  Lightweight logic (factory lookup, config parsing) is
tested without any image I/O.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from habit.core.preprocessing.preprocessor_factory import PreprocessorFactory

# Register steps whose modules are not imported from ``preprocessing.__init__``.
import habit.core.preprocessing.load_image  # noqa: F401

# Mimic a multi-sequence setup (e.g. brain MRI): native T1, T2, and post-contrast T1.
# First entry is the typical within-subject reference for registration.
_MODALITY_KEYS: List[str] = ["t1", "t2", "t1ce"]

# ---------------------------------------------------------------------------
# PreprocessorFactory
# ---------------------------------------------------------------------------

# Minimum constructor kwargs for factory smoke tests (matches real signatures;
# ``fixed_image`` is a data-dict key for the reference volume, not a path).
_MINIMAL_FACTORY_KWARGS: Dict[str, Dict[str, Any]] = {
    "n4_correction": {"keys": _MODALITY_KEYS},
    "histogram_standardization": {"keys": _MODALITY_KEYS},
    "zscore_normalization": {"keys": _MODALITY_KEYS},
    "resample": {"keys": _MODALITY_KEYS, "target_spacing": (1.0, 1.0, 1.0)},
    "registration": {"keys": _MODALITY_KEYS, "fixed_image": "t1"},
    "load_image": {"keys": _MODALITY_KEYS},
    "adaptive_histogram_equalization": {"keys": _MODALITY_KEYS},
}


class TestPreprocessorFactory:
    """Verify that the factory returns the right classes for known step names."""

    KNOWN_STEPS: List[str] = list(_MINIMAL_FACTORY_KWARGS.keys())

    @pytest.mark.parametrize("step_name", KNOWN_STEPS)
    def test_factory_returns_object_for_known_step(self, step_name: str) -> None:
        """Factory must not raise for any registered step name."""
        kwargs = _MINIMAL_FACTORY_KWARGS[step_name]
        preprocessor = PreprocessorFactory.create(step_name, **kwargs)
        assert preprocessor is not None

    def test_factory_raises_for_unknown_step(self) -> None:
        with pytest.raises(ValueError):
            PreprocessorFactory.create("nonexistent_step_xyz")

    def test_registration_simpleitk_backend_instantiation(self) -> None:
        """``RegistrationPreprocessor`` must construct when ``backend='simpleitk'``."""
        step = PreprocessorFactory.create(
            "registration",
            keys=_MODALITY_KEYS,
            fixed_image="t1",
            backend="simpleitk",
        )
        assert step is not None


# ---------------------------------------------------------------------------
# N4 bias-field correction (requires SimpleITK)
# ---------------------------------------------------------------------------


class TestN4Correction:
    sitk = pytest.importorskip("SimpleITK", reason="SimpleITK not installed")

    def test_n4_correction_instantiation(self) -> None:
        from habit.core.preprocessing.n4_correction import N4BiasFieldCorrection

        step = N4BiasFieldCorrection(keys=_MODALITY_KEYS)
        assert step is not None

    def test_n4_correction_is_callable(self) -> None:
        from habit.core.preprocessing.n4_correction import N4BiasFieldCorrection

        step = N4BiasFieldCorrection(keys=_MODALITY_KEYS)
        assert callable(step)


# ---------------------------------------------------------------------------
# Resample (requires SimpleITK)
# ---------------------------------------------------------------------------


class TestResample:
    sitk = pytest.importorskip("SimpleITK", reason="SimpleITK not installed")

    def test_resample_instantiation(self) -> None:
        from habit.core.preprocessing.resample import ResamplePreprocessor

        step = ResamplePreprocessor(
            keys=_MODALITY_KEYS,
            target_spacing=(1.0, 1.0, 1.0),
        )
        assert step is not None


# ---------------------------------------------------------------------------
# ZScore normalisation (numpy only, no heavy deps)
# ---------------------------------------------------------------------------


class TestZScoreNormalization:
    def test_zscore_instantiation(self) -> None:
        pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
        from habit.core.preprocessing.zscore_normalization import ZScoreNormalization

        step = ZScoreNormalization(keys=_MODALITY_KEYS)
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
        pytest.importorskip("SimpleITK", reason="SimpleITK not installed")
        from habit.core.preprocessing.base_preprocessor import BasePreprocessor

        abstract_methods = getattr(BasePreprocessor, "__abstractmethods__", set())
        assert len(abstract_methods) > 0, "BasePreprocessor should declare abstract methods"


if __name__ == "__main__":
    TestPreprocessorFactory().test_factory_returns_object_for_known_step("n4_correction")
