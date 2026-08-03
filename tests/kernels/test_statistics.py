# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for the L0 evaluation-statistics kernels.

The strongest equivalence evidence available: the kernels claim numerical
identity with the established v0.1 routines, so several tests compare the
two implementations directly on the same random inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from habit.kernels.icc import icc2_1, icc3_1, two_way_mean_squares
from habit.kernels.statistics import (
    compute_midrank,
    delong_roc_ci,
    delong_roc_test,
    delong_roc_variance,
    hosmer_lemeshow_test,
    spiegelhalter_z_test,
)


def _binary_problem(n: int = 60, seed: int = 0, separation: float = 1.5):
    """Return (y, scores) with a tunable signal-to-noise ratio."""
    rng = np.random.RandomState(seed)
    y = np.repeat([0, 1], n // 2)
    scores = rng.normal(size=n) * 0.5 + np.where(y == 1, separation, 0.0)
    # Squash into (0, 1) so the vectors double as probabilities.
    probabilities = 1.0 / (1.0 + np.exp(-(scores - scores.mean())))
    return y, probabilities


@pytest.mark.unit
def test_compute_midrank_handles_ties_one_based() -> None:
    """Tied values share the average rank; ranks are 1-based."""
    midranks = compute_midrank(np.array([0.0, 0.0, 1.0, 3.0, 3.0, 3.0]))
    np.testing.assert_allclose(midranks, [1.5, 1.5, 3.0, 5.0, 5.0, 5.0])


@pytest.mark.unit
def test_delong_auc_matches_sklearn() -> None:
    """The kernel AUC is sklearn's roc_auc_score."""
    from sklearn.metrics import roc_auc_score

    y, scores = _binary_problem()
    auc, variance = delong_roc_variance(y, scores)
    assert auc == pytest.approx(roc_auc_score(y, scores))
    assert variance > 0


@pytest.mark.unit
def test_delong_auc_matches_v0_1_implementation() -> None:
    """Kernel and v0.1 agree on AUC, variance, p-value and CI."""
    from habit.core.machine_learning.statistics.delong_test import (
        delong_roc_ci as v01_ci,
    )
    from habit.core.machine_learning.statistics.delong_test import (
        delong_roc_test as v01_test,
    )
    from habit.core.machine_learning.statistics.delong_test import (
        delong_roc_variance as v01_variance,
    )

    y, scores_a = _binary_problem(seed=1)
    _, scores_b = _binary_problem(seed=2, separation=0.8)
    assert delong_roc_variance(y, scores_a) == pytest.approx(v01_variance(y, scores_a))
    assert delong_roc_test(y, scores_a, scores_b) == pytest.approx(
        v01_test(y, scores_a, scores_b)
    )
    auc, ci = delong_roc_ci(y, scores_a)
    v01_auc, v01_ci_value = v01_ci(y, scores_a)
    assert auc == pytest.approx(v01_auc)
    np.testing.assert_allclose(ci, v01_ci_value)


@pytest.mark.unit
def test_delong_test_identical_scores_give_p_one() -> None:
    """A model compared with itself cannot differ."""
    y, scores = _binary_problem()
    assert delong_roc_test(y, scores, scores) == pytest.approx(1.0)


@pytest.mark.unit
def test_delong_rejects_non_binary_ground_truth() -> None:
    """Single-class or non-binary labels are a hard error."""
    with pytest.raises(ValueError):
        delong_roc_variance(np.ones(10), np.linspace(0, 1, 10))
    with pytest.raises(ValueError):
        delong_roc_test(
            np.repeat([0, 1, 2], 4), np.linspace(0, 1, 12), np.linspace(0, 1, 12)
        )


@pytest.mark.unit
def test_delong_ci_brackets_auc_and_clips_at_one() -> None:
    """The CI contains the point estimate and never exceeds 1."""
    # Moderate separation: at near-perfect separation the DeLong variance
    # degenerates (a known property of the estimator, shared with v0.1).
    y, scores = _binary_problem(separation=1.5)
    auc, ci = delong_roc_ci(y, scores, alpha=0.95)
    assert ci[0] <= auc <= ci[1]
    assert ci[1] <= 1.0


@pytest.mark.unit
def test_hosmer_lemeshow_matches_v0_1_implementation() -> None:
    """The kernel reproduces the v0.1 qcut-based statistic and p-value."""
    from habit.core.machine_learning.statistics.hosmer_lemeshow_test import (
        hosmer_lemeshow_test as v01_hl,
    )

    y, probabilities = _binary_problem(n=100, seed=3, separation=1.0)
    statistic, p_value = hosmer_lemeshow_test(y, probabilities, n_groups=10)
    data = pd.DataFrame({"y_true": y, "y_pred_proba": probabilities})
    v01_statistic, v01_p = v01_hl(data, Q=10)
    assert statistic == pytest.approx(v01_statistic)
    assert p_value == pytest.approx(v01_p)


@pytest.mark.unit
def test_hosmer_lemeshow_well_calibrated_model_passes() -> None:
    """Outcomes sampled from the predicted probabilities give a high p."""
    rng = np.random.RandomState(7)
    probabilities = rng.uniform(0.05, 0.95, size=400)
    y = (rng.uniform(size=400) < probabilities).astype(float)
    _, p_value = hosmer_lemeshow_test(y, probabilities, n_groups=10)
    assert p_value > 0.05


@pytest.mark.unit
def test_hosmer_lemeshow_input_validation() -> None:
    """Misaligned, non-binary, out-of-range or degenerate inputs raise."""
    y, probabilities = _binary_problem()
    with pytest.raises(ValueError):
        hosmer_lemeshow_test(y[:-1], probabilities)
    with pytest.raises(ValueError):
        hosmer_lemeshow_test(np.repeat([0.0, 2.0], 30), probabilities)
    with pytest.raises(ValueError):
        hosmer_lemeshow_test(y, probabilities + 2.0)
    with pytest.raises(ValueError):
        hosmer_lemeshow_test(y, probabilities, n_groups=1)
    with pytest.raises(ValueError):
        # Too many tied probabilities to form ten groups.
        hosmer_lemeshow_test(y, np.full_like(probabilities, 0.5))


@pytest.mark.unit
def test_spiegelhalter_matches_v0_1_implementation() -> None:
    """The kernel reproduces the v0.1 z statistic and p-value."""
    from habit.core.machine_learning.statistics.spiegelhalter_z_test import (
        spiegelhalter_z_test as v01_spiegelhalter,
    )

    y, probabilities = _binary_problem(seed=5, separation=1.0)
    z, p_value = spiegelhalter_z_test(y, probabilities)
    v01_z, v01_p = v01_spiegelhalter(y, probabilities)
    assert z == pytest.approx(v01_z)
    assert p_value == pytest.approx(v01_p)


@pytest.mark.unit
def test_spiegelhalter_formula_and_validation() -> None:
    """z follows sum(y - p) / sqrt(sum(p (1 - p))) exactly."""
    y = np.array([0.0, 1.0, 1.0, 0.0])
    probabilities = np.array([0.2, 0.8, 0.6, 0.4])
    expected_z = np.sum(y - probabilities) / np.sqrt(
        np.sum(probabilities * (1 - probabilities))
    )
    z, p_value = spiegelhalter_z_test(y, probabilities)
    assert z == pytest.approx(expected_z)
    assert 0.0 <= p_value <= 1.0
    with pytest.raises(ValueError):
        spiegelhalter_z_test(y[:-1], probabilities)


@pytest.mark.unit
def test_icc_kernels_match_pingouin() -> None:
    """ICC(2,1)/ICC(3,1) equal pingouin's ICC2/ICC3 single-measure rows."""
    pingouin = pytest.importorskip("pingouin")

    rng = np.random.RandomState(11)
    n_targets, k_raters = 12, 3
    target_effects = rng.normal(scale=2.0, size=(n_targets, 1))
    matrix = target_effects + rng.normal(scale=0.5, size=(n_targets, k_raters))
    long = pd.DataFrame(
        {
            "targets": np.repeat(np.arange(n_targets), k_raters),
            "raters": np.tile(np.arange(k_raters), n_targets),
            "ratings": matrix.ravel(),
        }
    )
    reference = pingouin.intraclass_corr(
        data=long, targets="targets", raters="raters", ratings="ratings"
    ).set_index("Type")
    # pingouin's row labels differ across versions: McGraw & Wong labels
    # ("ICC(A,1)" == ICC2, "ICC(C,1)" == ICC3) in >=0.6, Shrout & Fleiss labels
    # ("ICC2", "ICC3") in <=0.5. Accept either so the test is version-robust.
    icc2_label = "ICC(A,1)" if "ICC(A,1)" in reference.index else "ICC2"
    icc3_label = "ICC(C,1)" if "ICC(C,1)" in reference.index else "ICC3"
    assert icc2_1(matrix) == pytest.approx(reference.loc[icc2_label, "ICC"], abs=1e-10)
    assert icc3_1(matrix) == pytest.approx(reference.loc[icc3_label, "ICC"], abs=1e-10)


@pytest.mark.unit
def test_icc_perfect_agreement_and_constant_data() -> None:
    """Identical sessions give ICC 1; a constant matrix gives 0."""
    rng = np.random.RandomState(13)
    column = rng.normal(size=(8, 1))
    identical = np.hstack([column, column])
    assert icc3_1(identical) == pytest.approx(1.0)
    assert icc2_1(identical) == pytest.approx(1.0)
    assert icc3_1(np.full((4, 2), 1.5)) == 0.0
    assert icc2_1(np.full((4, 2), 1.5)) == 0.0


@pytest.mark.unit
def test_icc_input_validation() -> None:
    """Non-matrix, undersized or NaN-carrying inputs raise."""
    with pytest.raises(ValueError):
        two_way_mean_squares(np.arange(6.0))
    with pytest.raises(ValueError):
        two_way_mean_squares(np.ones((1, 3)))
    with pytest.raises(ValueError):
        two_way_mean_squares(np.array([[1.0, np.nan], [2.0, 3.0]]))
