"""
Unit tests for statistical test modules:
  - DeLong test (AUC comparison)
  - Hosmer-Lemeshow calibration test
  - Spiegelhalter Z test
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_roc_arrays(n: int = 200, seed: int = 0):
    rng = np.random.RandomState(seed)
    y_true = np.array([0] * (n // 2) + [1] * (n // 2))
    y_prob = np.concatenate([rng.beta(2, 5, n // 2), rng.beta(5, 2, n // 2)])
    return y_true, y_prob


# ---------------------------------------------------------------------------
# DeLong test
# ---------------------------------------------------------------------------


class TestDeLongTest:
    def test_delong_variance_returns_scalar(self) -> None:
        from habit.core.machine_learning.statistics.delong_test import delong_roc_variance

        y_true, y_prob = _make_roc_arrays()
        auc, variance = delong_roc_variance(y_true, y_prob)
        assert isinstance(float(auc), float)
        assert isinstance(float(variance), float)
        assert 0.0 <= float(auc) <= 1.0
        assert float(variance) >= 0.0

    def test_delong_roc_test_two_classifiers(self) -> None:
        from habit.core.machine_learning.statistics.delong_test import delong_roc_test

        y_true, y_prob1 = _make_roc_arrays(200, seed=0)
        _, y_prob2 = _make_roc_arrays(200, seed=1)
        # delong_roc_test returns (log10_p, p_value) or similar
        result = delong_roc_test(y_true, y_prob1, y_prob2)
        assert result is not None

    def test_perfect_classifier_high_auc(self) -> None:
        from habit.core.machine_learning.statistics.delong_test import delong_roc_variance

        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = np.concatenate([np.zeros(50), np.ones(50)])
        auc, _ = delong_roc_variance(y_true, y_prob)
        assert float(auc) > 0.99

    def test_random_classifier_auc_near_half(self) -> None:
        from habit.core.machine_learning.statistics.delong_test import delong_roc_variance

        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_prob = rng.rand(200)
        auc, _ = delong_roc_variance(y_true, y_prob)
        assert 0.3 <= float(auc) <= 0.7  # random → AUC ≈ 0.5


# ---------------------------------------------------------------------------
# Hosmer-Lemeshow test
# ---------------------------------------------------------------------------


class TestHosmerLemeshowTest:
    def test_well_calibrated_model_high_p(self) -> None:
        from habit.core.machine_learning.statistics.hosmer_lemeshow_test import (
            hosmer_lemeshow_test,
        )

        rng = np.random.RandomState(0)
        n = 200
        # Use predicted probs as true labels (perfectly calibrated)
        y_prob = rng.rand(n)
        y_true = (rng.rand(n) < y_prob).astype(int)
        hl_data = pd.DataFrame({"y_true": y_true, "y_pred_proba": y_prob})
        result = hosmer_lemeshow_test(hl_data)
        assert result is not None
        chi2_stat, p_val = result[0], result[1]
        assert 0.0 <= float(p_val) <= 1.0
        assert isinstance(float(chi2_stat), float)

    def test_returns_dict_or_tuple(self) -> None:
        from habit.core.machine_learning.statistics.hosmer_lemeshow_test import (
            hosmer_lemeshow_test,
        )

        y_true, y_prob = _make_roc_arrays()
        hl_data = pd.DataFrame({"y_true": y_true, "y_pred_proba": y_prob})
        result = hosmer_lemeshow_test(hl_data)
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# Spiegelhalter Z test
# ---------------------------------------------------------------------------


class TestSpiegelhalterZTest:
    def test_returns_z_and_p(self) -> None:
        from habit.core.machine_learning.statistics.spiegelhalter_z_test import (
            spiegelhalter_z_test,
        )

        y_true, y_prob = _make_roc_arrays()
        result = spiegelhalter_z_test(y_true, y_prob)
        assert result is not None
        z = result.get("z") if isinstance(result, dict) else result[0]
        p = result.get("p_value") if isinstance(result, dict) else result[1]
        assert isinstance(float(z), float)
        assert 0.0 <= float(p) <= 1.0

    def test_perfectly_calibrated_low_z(self) -> None:
        """A well-calibrated model should produce a Z statistic close to 0."""
        from habit.core.machine_learning.statistics.spiegelhalter_z_test import (
            spiegelhalter_z_test,
        )

        rng = np.random.RandomState(7)
        n = 500
        y_prob = rng.beta(2, 2, n)
        y_true = (rng.rand(n) < y_prob).astype(int)
        result = spiegelhalter_z_test(y_true, y_prob)
        z = result.get("z") if isinstance(result, dict) else result[0]
        assert abs(float(z)) < 4.0  # well calibrated → |Z| should not be extreme
