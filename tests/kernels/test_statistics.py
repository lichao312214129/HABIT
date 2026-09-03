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
"""Tests for the L0 evaluation-statistics kernels."""

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
def test_statistics_match_frozen_legacy_reference_values() -> None:
    """
    Preserve the v0.1 DeLong and calibration statistics after compat removal.

    The constants were recorded from the former implementations on this
    deterministic fixture before their source modules were deleted. Keeping
    the reference values in this kernel-only test prevents a deleted compat
    module from becoming the test oracle again.
    """
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    scores_a = np.array([0.05, 0.91, 0.20, 0.80, 0.42, 0.77, 0.31, 0.65, 0.73, 0.56])
    scores_b = np.array([0.12, 0.74, 0.11, 0.62, 0.38, 0.69, 0.47, 0.54, 0.66, 0.51])

    auc, variance = delong_roc_variance(y, scores_a)
    p_value = delong_roc_test(y, scores_a, scores_b)
    ci_auc, ci = delong_roc_ci(y, scores_a)
    hl_statistic, hl_p_value = hosmer_lemeshow_test(y, scores_a, n_groups=10)
    z_statistic, z_p_value = spiegelhalter_z_test(y, scores_a)

    assert auc == pytest.approx(0.91999996)
    assert variance == pytest.approx(0.008799998950958282)
    assert p_value == pytest.approx(0.47950066)
    assert ci_auc == pytest.approx(0.91999996)
    np.testing.assert_allclose(ci, [0.73613905, 1.0], rtol=1e-7, atol=1e-8)
    assert hl_statistic == pytest.approx(6.151526797782616)
    assert hl_p_value == pytest.approx(0.6302630640589141)
    assert z_statistic == pytest.approx(-0.30194054243855895)
    assert z_p_value == pytest.approx(0.7626973888423847)


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
def test_icc2_1_estimate_matches_pingouin_including_ci() -> None:
    """ICC(2,1) value AND Satterthwaite CI equal pingouin's ICC(A,1) row.

    pingouin is the independent oracle for HABIT's own ICC kernels: the
    voxel-level kernels must stay dependency-free and reproduce the Prior
    2024 erratum formulas, which pingouin does not implement, so equality is
    asserted here instead of calling pingouin at run time. pingouin rounds
    its published CI, so its rounding option is disabled for the comparison.
    """
    pingouin = pytest.importorskip("pingouin")
    from habit.kernels.voxel_icc import icc2_1_estimate

    rng = np.random.RandomState(23)
    n_targets, k_raters = 40, 3
    target_effects = rng.normal(scale=3.0, size=(n_targets, 1))
    matrix = target_effects + rng.normal(scale=0.7, size=(n_targets, k_raters))
    # A systematic rater offset is what separates absolute agreement (model 2)
    # from consistency, so the delineation-axis formula is exercised.
    matrix = matrix + np.array([0.0, 1.5, -1.0])
    long = pd.DataFrame(
        {
            "targets": np.repeat(np.arange(n_targets), k_raters),
            "raters": np.tile(np.arange(k_raters), n_targets),
            "ratings": matrix.ravel(),
        }
    )
    # pingouin rounds CI95 through a column-specific option, not the global one.
    saved = {key: pingouin.options.get(key) for key in ("round", "round.column.CI95")}
    pingouin.options["round"] = None
    pingouin.options["round.column.CI95"] = None
    try:
        reference = pingouin.intraclass_corr(
            data=long, targets="targets", raters="raters", ratings="ratings"
        ).set_index("Type")
    finally:
        pingouin.options.update(saved)
    label = "ICC(A,1)" if "ICC(A,1)" in reference.index else "ICC2"
    row = reference.loc[label]
    lower, upper = (float(v) for v in row["CI95"])

    estimate = icc2_1_estimate(matrix)
    assert estimate.value == pytest.approx(float(row["ICC"]), abs=1e-10)
    assert estimate.lcl == pytest.approx(lower, abs=1e-8)
    assert estimate.ucl == pytest.approx(upper, abs=1e-8)


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
