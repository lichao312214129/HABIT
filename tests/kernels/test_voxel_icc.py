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
"""Tests for the L0 voxel-level ICC kernels.

The hand-computed reference matrix [[1, 2], [2, 1], [3, 3]] pins the
erratum-corrected formulas: MSR = 1.5, MSC = 0, MSE = 0.5, so
ICC(3A,1) = (1.5 - 0.5) / (1.5 + 0.5 + (2/3)(0/3 - 0.5)) = 0.6 and
ICC(3C,1) = (1.5 - 0.5) / (1.5 + 0.5) = 0.5, with F-ratio confidence
limits (FR = 3, F(0.975; 2, 2) = 39) giving LCL < 0 (clipped to 0) and
UCL = 116/118.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from habit.kernels.icc import icc2_1, icc3_1
from habit.kernels.voxel_icc import ICCEstimate, icc3a_1, icc3c_1

#: Hand-computed reference: MSR = 1.5, MSC = 0, MSE = 0.5 (see module docstring).
REFERENCE = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])


class TestHandComputedReference:
    def test_icc3a_1_value(self) -> None:
        estimate = icc3a_1(REFERENCE)
        assert estimate.value == pytest.approx(0.6, abs=1e-12)

    def test_icc3c_1_value(self) -> None:
        estimate = icc3c_1(REFERENCE)
        assert estimate.value == pytest.approx(0.5, abs=1e-12)

    def test_confidence_limits(self) -> None:
        f_crit = stats.f.ppf(0.975, 2, 2)  # 39.0
        f_lower, f_upper = 3.0 / f_crit, 3.0 * f_crit
        expected_lcl = max(0.0, (f_lower - 1.0) / (f_lower + 1.0))
        expected_ucl = (f_upper - 1.0) / (f_upper + 1.0)
        for estimate in (icc3a_1(REFERENCE), icc3c_1(REFERENCE)):
            assert estimate.lcl == pytest.approx(expected_lcl, abs=1e-12)
            assert estimate.ucl == pytest.approx(expected_ucl, abs=1e-12)
            assert estimate.lcl == 0.0  # raw LCL is negative, clipped


class TestDegenerateInputs:
    def test_identical_columns_varying_rows_is_perfect(self) -> None:
        column = np.linspace(0.0, 10.0, 50)
        data = np.column_stack([column, column, column])
        assert icc3a_1(data) == ICCEstimate(1.0, 1.0, 1.0)
        assert icc3c_1(data) == ICCEstimate(1.0, 1.0, 1.0)

    def test_constant_matrix_is_zero(self) -> None:
        data = np.full((10, 3), 7.0)
        assert icc3a_1(data) == ICCEstimate(0.0, 0.0, 0.0)
        assert icc3c_1(data) == ICCEstimate(0.0, 0.0, 0.0)

    def test_negative_value_truncated_at_zero(self) -> None:
        # Row variance far below the residual variance: raw ICC < 0.
        data = np.array([[1.0, 2.0], [2.0, 1.0], [1.4, 1.6], [1.6, 1.4]])
        estimate = icc3c_1(data)
        assert estimate.value == 0.0

    def test_nan_rejected(self) -> None:
        data = np.array([[1.0, np.nan], [2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN"):
            icc3a_1(data)

    def test_1d_rejected(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            icc3c_1(np.zeros(6))

    def test_single_column_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            icc3a_1(np.zeros((5, 1)))


class TestConsistencyWithExistingKernels:
    def test_icc3c_1_matches_icc3_1_point_estimate(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 1.0, size=(200, 3))
        data += np.linspace(0.0, 5.0, 200)[:, None]  # between-voxel signal
        assert icc3c_1(data).value == pytest.approx(icc3_1(data), abs=1e-12)

    def test_icc3a_1_approaches_icc2_1_for_large_n(self) -> None:
        # With n >> 1 the erratum correction term vanishes and ICC(3A,1)
        # coincides with the classical absolute-agreement formula.
        rng = np.random.default_rng(1)
        data = rng.normal(0.0, 0.5, size=(20000, 2))
        data += rng.normal(0.0, 3.0, size=(20000, 1))
        assert icc3a_1(data).value == pytest.approx(icc2_1(data), abs=1e-3)


class TestScreenBehaviour:
    def test_high_agreement_passes_lcl_screen(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.normal(0.0, 0.05, size=(1000, 2))
        data += rng.normal(0.0, 5.0, size=(1000, 1))
        estimate = icc3a_1(data)
        assert estimate.lcl >= 0.5
        assert estimate.lcl <= estimate.value <= estimate.ucl

    def test_pure_noise_fails_lcl_screen(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.normal(0.0, 1.0, size=(1000, 2))
        estimate = icc3a_1(data)
        assert estimate.lcl < 0.5

    def test_icc_tracks_theoretical_noise_mixture(self) -> None:
        """
        The point estimate follows the analytic noise-mixture law.

        For two conditions s + lam*n1 and s + lam*n2 with independent
        equal-variance noise draws, the rater-effect term of the erratum
        formula vanishes and ICC(3A,1) reduces to
        var(s) / (var(s) + lam^2 * var(n)). With var(s) = 4 and
        var(n) = 1 the expected values are 1.0, 0.941, 0.8, 0.5 and 0.2
        for lam = 0, 0.5, 1, 2, 4 -- and lam = 2 (noise variance equal to
        signal variance) lands exactly on the screen's 0.5 threshold.
        """
        rng = np.random.default_rng(7)
        n = 20000
        signal = rng.normal(0.0, 2.0, n)
        expected = {0.0: 1.0, 0.5: 4.0 / 4.25, 1.0: 0.8, 2.0: 0.5, 4.0: 0.2}
        estimates = []
        for lam, theory in expected.items():
            data = np.column_stack(
                [
                    signal + lam * rng.normal(0.0, 1.0, n),
                    signal + lam * rng.normal(0.0, 1.0, n),
                ]
            )
            value = icc3a_1(data).value
            assert value == pytest.approx(theory, abs=0.02)
            estimates.append(value)
        # The scale is calibrated: the estimate decreases monotonically as
        # the noise fraction grows.
        assert estimates == sorted(estimates, reverse=True)

    def test_alpha_widens_interval(self) -> None:
        rng = np.random.default_rng(4)
        data = rng.normal(0.0, 0.2, size=(500, 2))
        data += rng.normal(0.0, 2.0, size=(500, 1))
        narrow = icc3a_1(data, alpha=0.05)
        wide = icc3a_1(data, alpha=0.01)
        assert wide.lcl <= narrow.lcl
        assert wide.ucl >= narrow.ucl
