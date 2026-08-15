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
"""Tests for the L3 precision-analysis domain package.

The panel tests reuse the hand-computed reference matrix of the kernel
tests ([[1, 2], [2, 1], [3, 3]] -> ICC(3A,1) = 0.6, ICC(3C,1) = 0.5) to
pin the plumbing: voxel alignment, per-condition min-max scaling and
pairwise-complete NaN handling.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from habit.contracts import (
    ArrayImageRef,
    Geometry,
    HabitatMap,
    Subject,
    VoxelFeatureField,
)
from habit.domain.precision import (
    BSplineDeformPerturbation,
    GaussianNoisePerturbation,
    ImagePerturbationRegistry,
    PerturbationChain,
    PreciseFeatureSet,
    RigidPerturbation,
    RotationPerturbation,
    TranslationPerturbation,
    aggregate_panels,
    align_habitat_map,
    habitat_stability,
    identify_precise_features,
    precision_panel,
    prior2024_retest_perturbation,
)
from habit.exceptions import HABITAPIError, OptionalDependencyError
from habit.kernels.voxel_icc import icc3a_1, icc3c_1

from .conftest import make_habitat_map, make_subject, provenance


def _field(
    values: np.ndarray,
    feature_names: Tuple[str, ...] = ("f1",),
    *,
    shape: Tuple[int, int, int] = (4, 4, 4),
    subject_id: str = "P1",
    drop_last_voxel: bool = False,
) -> VoxelFeatureField:
    """Build a voxel field from a (n_voxels, n_features) matrix on a cubic grid."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    index = np.array(
        [(z, y, x) for z in range(shape[0]) for y in range(shape[1]) for x in range(shape[2])]
    )
    if drop_last_voxel:
        index = index[1 : values.shape[0] + 1]
    else:
        index = index[: values.shape[0]]
    return VoxelFeatureField(
        subject_id=subject_id,
        feature_names=tuple(feature_names),
        values=values,
        voxel_index=index,
        geometry=Geometry.from_array(shape),
        provenance=provenance(),
    )


def _blob_subject(shape: Tuple[int, int, int] = (12, 12, 12)) -> Subject:
    """Subject with a compact centred blob, so shifts never touch the boundary."""
    geometry = Geometry.from_array(shape)
    array = np.zeros(shape, dtype=np.float64)
    array[4:8, 4:8, 4:8] = 10.0
    mask = np.zeros(shape, dtype=np.int32)
    mask[2:10, 2:10, 2:10] = 1
    return Subject(
        subject_id="P1",
        images={"T1": ArrayImageRef(array=array, geometry=geometry)},
        masks={"tumor": ArrayImageRef(array=mask, geometry=geometry)},
    )


def _com_first_axis(array: np.ndarray) -> float:
    """Centre of mass along the first (z) array axis."""
    return float(
        (array * np.arange(array.shape[0])[:, None, None]).sum() / array.sum()
    )


def _panel(
    features: Tuple[str, ...], lcl: float, value: float = 0.9
) -> pd.DataFrame:
    """Build a cohort-panel-shaped frame with a uniform LCL."""
    return pd.DataFrame(
        {
            "value": value,
            "lcl": lcl,
            "ucl": min(1.0, value + 0.05),
            "n_voxels": 100,
        },
        index=pd.Index(list(features), name="feature"),
    )


class TestPerturbationRegistry:
    def test_builtins_registered(self) -> None:
        available = ImagePerturbationRegistry.available()
        assert "gaussian_noise" in available
        assert "translation" in available
        assert "rotation" in available
        assert "rigid" in available
        assert "bspline_deform" in available

    def test_create_validates_params(self) -> None:
        component = ImagePerturbationRegistry.create("rotation", angle_degrees=1.5)
        assert isinstance(component, RotationPerturbation)
        assert component.angle_degrees == 1.5
        with pytest.raises(Exception):
            ImagePerturbationRegistry.create("rotation", bogus_param=1)


class TestGaussianNoisePerturbation:
    def test_adds_noise_and_preserves_original(self) -> None:
        subject = make_subject("P1")
        original = subject.image("T1").data.copy()
        perturbed = GaussianNoisePerturbation(sigma=0.5)(
            subject, rng=np.random.default_rng(0)
        )
        assert isinstance(perturbed, Subject)
        assert perturbed is not subject
        np.testing.assert_array_equal(subject.image("T1").data, original)
        moved = np.asarray(perturbed.image("T1").data)
        assert moved.shape == original.shape
        assert not np.allclose(moved, original)
        # Masks are untouched by an intensity perturbation.
        np.testing.assert_array_equal(
            np.asarray(perturbed.mask("tumor").data),
            np.asarray(subject.mask("tumor").data),
        )

    def test_seeded_is_reproducible(self) -> None:
        subject = make_subject("P1")
        step = GaussianNoisePerturbation(sigma=0.5)
        first = step(subject, rng=np.random.default_rng(7))
        second = step(subject, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(
            np.asarray(first.image("T1").data), np.asarray(second.image("T1").data)
        )

    def test_zero_sigma_is_identity(self) -> None:
        subject = make_subject("P1")
        perturbed = GaussianNoisePerturbation(sigma=0.0)(
            subject, rng=np.random.default_rng(0)
        )
        np.testing.assert_allclose(
            np.asarray(perturbed.image("T1").data),
            np.asarray(subject.image("T1").data),
        )

    def test_estimated_sigma(self) -> None:
        subject = make_subject("P1")
        perturbed = GaussianNoisePerturbation(noise_method="chang")(
            subject, rng=np.random.default_rng(0)
        )
        assert not np.allclose(
            np.asarray(perturbed.image("T1").data),
            np.asarray(subject.image("T1").data),
        )

    def test_roi_std_uses_mask(self) -> None:
        subject = make_subject("P1")
        perturbed = GaussianNoisePerturbation(noise_method="roi_std")(
            subject, rng=np.random.default_rng(0)
        )
        assert isinstance(perturbed, Subject)

    def test_bad_method_raises(self) -> None:
        with pytest.raises(HABITAPIError, match="noise_method"):
            GaussianNoisePerturbation(noise_method="bogus")


class TestTranslationPerturbation:
    def test_fixed_shift_moves_image_and_mask(self) -> None:
        subject = _blob_subject()
        perturbed = TranslationPerturbation(
            shift_voxels=(0.0, 0.0, 1.0), interpolator="linear"
        )(subject, rng=np.random.default_rng(0))
        before = np.asarray(subject.image("T1").data)
        after = np.asarray(perturbed.image("T1").data)
        # shift_voxels is (x, y, z): a +1 z shift moves content along the
        # FIRST array axis by +1 voxel.
        assert _com_first_axis(after) == pytest.approx(
            _com_first_axis(before) + 1.0, abs=1e-6
        )
        # The mask moved too, with integer labels preserved.
        mask_after = np.asarray(perturbed.mask("tumor").data)
        assert set(np.unique(mask_after).tolist()) <= {0, 1}
        assert _com_first_axis(mask_after) == pytest.approx(
            _com_first_axis(np.asarray(subject.mask("tumor").data)) + 1.0, abs=1e-6
        )

    def test_random_shift_is_seeded(self) -> None:
        subject = make_subject("P1")
        step = TranslationPerturbation(max_shift_voxels=1.0)
        first = step(subject, rng=np.random.default_rng(3))
        second = step(subject, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(
            np.asarray(first.image("T1").data), np.asarray(second.image("T1").data)
        )

    def test_bad_shift_raises(self) -> None:
        with pytest.raises(HABITAPIError, match="3 components"):
            TranslationPerturbation(shift_voxels=(1.0, 2.0))

    def test_shift_fraction_is_subvoxel(self) -> None:
        subject = _blob_subject()
        perturbed = TranslationPerturbation(
            shift_fraction=0.5, random_signs=False, interpolator="linear"
        )(subject, rng=np.random.default_rng(0))
        before = np.asarray(subject.image("T1").data)
        after = np.asarray(perturbed.image("T1").data)
        # +0.5 voxel along every SimpleITK axis; z is array axis 0.
        assert _com_first_axis(after) == pytest.approx(
            _com_first_axis(before) + 0.5, abs=0.05
        )

    def test_shift_voxels_and_fraction_conflict(self) -> None:
        with pytest.raises(HABITAPIError, match="not both"):
            TranslationPerturbation(shift_voxels=(0.1, 0.0, 0.0), shift_fraction=0.5)


class TestRotationPerturbation:
    def test_zero_angle_is_identity(self) -> None:
        subject = make_subject("P1")
        perturbed = RotationPerturbation(angle_degrees=0.0, interpolator="linear")(
            subject, rng=np.random.default_rng(0)
        )
        np.testing.assert_allclose(
            np.asarray(perturbed.image("T1").data),
            np.asarray(subject.image("T1").data),
            atol=1e-12,
        )

    def test_deterministic_rng_not_consumed(self) -> None:
        subject = make_subject("P1")
        step = RotationPerturbation(angle_degrees=0.5)
        first = step(subject, rng=np.random.default_rng(1))
        second = step(subject, rng=np.random.default_rng(999))
        np.testing.assert_array_equal(
            np.asarray(first.image("T1").data), np.asarray(second.image("T1").data)
        )

    def test_bad_axis_raises(self) -> None:
        with pytest.raises(HABITAPIError, match="axis"):
            RotationPerturbation(axis="w")


class TestRigidPerturbation:
    def test_zero_motion_is_identity(self) -> None:
        subject = make_subject("P1")
        perturbed = RigidPerturbation(
            shift_voxels=(0.0, 0.0, 0.0),
            angle_degrees=0.0,
            interpolator="linear",
        )(subject, rng=np.random.default_rng(0))
        np.testing.assert_allclose(
            np.asarray(perturbed.image("T1").data),
            np.asarray(subject.image("T1").data),
            atol=1e-12,
        )

    def test_create_from_registry(self) -> None:
        component = ImagePerturbationRegistry.create(
            "rigid", shift_fraction=0.25, angle_degrees=0.5
        )
        assert isinstance(component, RigidPerturbation)
        assert component.shift_fraction == 0.25


class TestPrior2024RetestPerturbation:
    def test_paper_chain_has_three_steps(self) -> None:
        chain = prior2024_retest_perturbation()
        assert len(chain.steps) == 3
        assert isinstance(chain.steps[0], GaussianNoisePerturbation)
        assert isinstance(chain.steps[1], TranslationPerturbation)
        assert isinstance(chain.steps[2], RotationPerturbation)
        assert chain.steps[1].shift_fraction == 0.5
        assert chain.steps[2].angle_degrees == 0.5
        assert not any(
            isinstance(step, BSplineDeformPerturbation) for step in chain.steps
        )

    def test_single_resample_uses_rigid(self) -> None:
        chain = prior2024_retest_perturbation(single_resample=True)
        assert len(chain.steps) == 2
        assert isinstance(chain.steps[1], RigidPerturbation)

    def test_runs_on_blob_subject(self) -> None:
        subject = _blob_subject()
        perturbed = prior2024_retest_perturbation()(
            subject, rng=np.random.default_rng(0)
        )
        assert perturbed.image("T1").data.shape == subject.image("T1").data.shape


class TestBSplineDeformPerturbation:
    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(HABITAPIError, match="low must be"):
            BSplineDeformPerturbation(sigma_range=(8.0, 5.0))

    def test_create_from_registry(self) -> None:
        component = ImagePerturbationRegistry.create(
            "bspline_deform", magnitude_range=(1.0, 2.0)
        )
        assert isinstance(component, BSplineDeformPerturbation)
        assert component.magnitude_range == (1.0, 2.0)

    def _sphere_subject(self, side: int = 32) -> Subject:
        """Subject whose ROI is an interior sphere so a warp can flip voxels."""
        subject = make_subject("P1", shape=(side, side, side))
        centre = side / 2.0
        radius = side / 4.0
        zz, yy, xx = np.ogrid[:side, :side, :side]
        ball = (
            (zz - centre) ** 2 + (yy - centre) ** 2 + (xx - centre) ** 2
        ) <= radius**2
        geometry = subject.mask("tumor").geometry
        return Subject(
            subject_id=subject.subject_id,
            images=subject.images,
            masks={
                "tumor": ArrayImageRef(
                    array=ball.astype(np.int32), geometry=geometry
                )
            },
        )

    def test_warps_image_and_mask_on_same_grid(self) -> None:
        subject = self._sphere_subject(32)
        step = BSplineDeformPerturbation(
            sigma_range=(1.5, 2.5),
            magnitude_range=(4.0, 6.0),
            device="cpu",
        )
        try:
            perturbed = step(subject, rng=np.random.default_rng(0))
        except OptionalDependencyError:
            pytest.skip("monai extra is not installed")
        before_image = np.asarray(subject.image("T1").data)
        after_image = np.asarray(perturbed.image("T1").data)
        before_mask = np.asarray(subject.mask("tumor").data)
        after_mask = np.asarray(perturbed.mask("tumor").data)
        assert after_image.shape == before_image.shape
        assert after_mask.shape == before_mask.shape
        assert after_image.dtype == np.float64
        assert set(np.unique(after_mask).tolist()) <= {0, 1}
        assert not np.array_equal(after_mask, before_mask)
        assert not np.allclose(after_image, before_image)

    def test_same_seed_is_reproducible(self) -> None:
        subject = self._sphere_subject(24)
        step = BSplineDeformPerturbation(
            sigma_range=(1.5, 2.0),
            magnitude_range=(3.0, 4.0),
            device="cpu",
        )
        try:
            first = step(subject, rng=np.random.default_rng(11))
            second = step(subject, rng=np.random.default_rng(11))
        except OptionalDependencyError:
            pytest.skip("monai extra is not installed")
        np.testing.assert_allclose(
            np.asarray(first.image("T1").data),
            np.asarray(second.image("T1").data),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            np.asarray(first.mask("tumor").data),
            np.asarray(second.mask("tumor").data),
        )

    def test_target_dice_scales_warp_to_requested_roi_overlap(self) -> None:
        """A frozen MONAI field can be scaled so ROI Dice hits a target."""
        from habit.kernels.image_perturbation import binary_mask_dice

        subject = self._sphere_subject(32)
        step = BSplineDeformPerturbation(
            sigma_range=(1.5, 2.5),
            magnitude_range=(4.0, 6.0),
            target_dice=0.90,
            dice_tolerance=0.04,
            device="cpu",
        )
        try:
            perturbed = step(subject, rng=np.random.default_rng(3))
        except OptionalDependencyError:
            pytest.skip("monai extra is not installed")
        dice = binary_mask_dice(
            np.asarray(subject.mask("tumor").data),
            np.asarray(perturbed.mask("tumor").data),
        )
        assert dice == pytest.approx(0.90, abs=0.04)


class TestPerturbationChain:
    def test_empty_chain_raises(self) -> None:
        with pytest.raises(HABITAPIError, match="at least one step"):
            PerturbationChain([])

    def test_steps_apply_in_order(self) -> None:
        subject = _blob_subject()
        chain = PerturbationChain(
            [
                TranslationPerturbation(shift_voxels=(0.0, 0.0, 0.5), interpolator="linear"),
                TranslationPerturbation(shift_voxels=(0.0, 0.0, 0.5), interpolator="linear"),
            ]
        )
        perturbed = chain(subject, rng=np.random.default_rng(0))
        before = np.asarray(subject.image("T1").data)
        after = np.asarray(perturbed.image("T1").data)
        assert _com_first_axis(after) == pytest.approx(
            _com_first_axis(before) + 1.0, abs=1e-6
        )

    def test_spec_records_steps(self) -> None:
        chain = PerturbationChain(
            [GaussianNoisePerturbation(sigma=1.0), RotationPerturbation()]
        )
        payload = chain.spec.to_dict()
        assert payload["name"] == "perturbation_chain"
        assert [step["name"] for step in payload["params"]["steps"]] == [
            "gaussian_noise",
            "rotation",
        ]


class TestPrecisionPanel:
    def test_identical_conditions_are_perfect(self) -> None:
        values = np.arange(20, dtype=np.float64)
        field = _field(values)
        panel = precision_panel({"a": field, "b": field})
        assert panel.loc["f1", "value"] == 1.0
        assert panel.loc["f1", "lcl"] == 1.0

    def test_hand_computed_reference_unscaled(self) -> None:
        first = _field(np.array([1.0, 2.0, 3.0]))
        second = _field(np.array([2.0, 1.0, 3.0]))
        panel = precision_panel(
            {"a": first, "b": second}, agreement="absolute", scale=False, min_voxels=2
        )
        assert panel.loc["f1", "value"] == pytest.approx(0.6, abs=1e-12)
        panel_c = precision_panel(
            {"a": first, "b": second}, agreement="consistency", scale=False, min_voxels=2
        )
        assert panel_c.loc["f1", "value"] == pytest.approx(0.5, abs=1e-12)

    def test_scaling_removes_condition_scale(self) -> None:
        # Same pattern on a different scale: identical after per-condition
        # min-max scaling, strongly penalised without it.
        first = _field(np.array([1.0, 2.0, 3.0]))
        second = _field(np.array([15.0, 25.0, 35.0]))
        scaled_panel = precision_panel(
            {"a": first, "b": second}, agreement="absolute", scale=True, min_voxels=2
        )
        assert scaled_panel.loc["f1", "value"] == pytest.approx(1.0, abs=1e-12)
        unscaled_panel = precision_panel(
            {"a": first, "b": second}, agreement="absolute", scale=False, min_voxels=2
        )
        assert unscaled_panel.loc["f1", "value"] < 0.2

    def test_scaling_matches_kernel_on_scaled_matrix(self) -> None:
        first = _field(np.array([1.0, 2.0, 3.0]))
        second = _field(np.array([2.0, 1.0, 3.0]))
        panel = precision_panel(
            {"a": first, "b": second}, agreement="absolute", scale=True, min_voxels=2
        )
        # Per-condition min-max: [1,2,3] -> [0,0.5,1], [2,1,3] -> [0.5,0,1].
        scaled = np.array([[0.0, 0.5], [0.5, 0.0], [1.0, 1.0]])
        expected = icc3a_1(scaled)
        assert panel.loc["f1", "value"] == pytest.approx(expected.value, abs=1e-12)

    def test_nan_rows_dropped_pairwise(self) -> None:
        first = _field(np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]))
        second = _field(np.array([1.1, 2.0, 3.1, 4.0, 5.2, 6.0, 7.1, 8.0, 9.2, 10.0, 11.1]))
        panel = precision_panel({"a": first, "b": second}, scale=False)
        assert panel.loc["f1", "n_voxels"] == 10
        assert panel.loc["f1", "value"] > 0.99

    def test_too_few_voxels_is_nan(self) -> None:
        first = _field(np.array([1.0, 2.0, 3.0]))
        second = _field(np.array([2.0, 1.0, 3.0]))
        panel = precision_panel({"a": first, "b": second})
        assert np.isnan(panel.loc["f1", "value"])
        assert panel.loc["f1", "n_voxels"] == 3

    def test_voxel_alignment_intersects(self) -> None:
        values_a = np.arange(1.0, 12.0)
        values_b = np.arange(1.0, 11.0) + 0.1
        first = _field(values_a)
        second = _field(values_b, drop_last_voxel=True)
        panel = precision_panel({"a": first, "b": second}, scale=False)
        # The second field lacks the first voxel; 10 common voxels remain.
        assert panel.loc["f1", "n_voxels"] == 10
        assert panel.loc["f1", "value"] > 0.99

    def test_single_condition_raises(self) -> None:
        with pytest.raises(HABITAPIError, match="at least two"):
            precision_panel({"a": _field(np.arange(5.0))})

    def test_feature_mismatch_raises(self) -> None:
        first = _field(np.arange(5.0), ("f1",))
        second = _field(np.arange(5.0), ("f2",))
        with pytest.raises(HABITAPIError, match="features differ"):
            precision_panel({"a": first, "b": second})

    def test_bad_agreement_raises(self) -> None:
        field = _field(np.arange(12.0))
        with pytest.raises(HABITAPIError, match="agreement"):
            precision_panel({"a": field, "b": field}, agreement="bogus")


class TestAggregatePanels:
    def test_median_across_subjects(self) -> None:
        panels = [
            _panel(("f1", "f2"), lcl=value)
            for value in (0.4, 0.6, 0.8)
        ]
        cohort = aggregate_panels(panels)
        assert cohort.loc["f1", "lcl"] == pytest.approx(0.6)

    def test_nan_does_not_veto(self) -> None:
        good = _panel(("f1",), lcl=0.8)
        bad = _panel(("f1",), lcl=np.nan)
        cohort = aggregate_panels([good, bad])
        assert cohort.loc["f1", "lcl"] == pytest.approx(0.8)

    def test_index_mismatch_raises(self) -> None:
        with pytest.raises(HABITAPIError, match="feature index"):
            aggregate_panels([_panel(("f1",), 0.5), _panel(("f2",), 0.5)])

    def test_empty_raises(self) -> None:
        with pytest.raises(HABITAPIError, match="at least one"):
            aggregate_panels([])


class TestIdentifyPreciseFeatures:
    def test_threshold_applied_in_every_experiment(self) -> None:
        experiments = {
            "repeatability": pd.DataFrame(
                {"value": 0.9, "lcl": [0.6, 0.4, 0.9], "ucl": 0.95, "n_voxels": 100},
                index=pd.Index(["f1", "f2", "f3"], name="feature"),
            ),
            "reproducibility": pd.DataFrame(
                {"value": 0.9, "lcl": [0.7, 0.8, 0.9], "ucl": 0.95, "n_voxels": 100},
                index=pd.Index(["f1", "f2", "f3"], name="feature"),
            ),
        }
        result = identify_precise_features(experiments, lcl_threshold=0.5)
        assert isinstance(result, PreciseFeatureSet)
        # f2 fails repeatability; f1 and f3 pass both.
        assert result.feature_names == ("f1", "f3")
        assert result.experiments == ("repeatability", "reproducibility")

    def test_expert_overrides(self) -> None:
        experiments = {
            "repeatability": pd.DataFrame(
                {"value": 0.9, "lcl": [0.6, 0.4, 0.9], "ucl": 0.95, "n_voxels": 100},
                index=pd.Index(["f1", "f2", "f3"], name="feature"),
            )
        }
        result = identify_precise_features(
            experiments, lcl_threshold=0.5, include=("f2",), exclude=("f3",)
        )
        assert result.feature_names == ("f1", "f2")

    def test_unknown_override_raises(self) -> None:
        experiments = {"repeatability": _panel(("f1",), 0.9)}
        with pytest.raises(HABITAPIError, match="unknown features"):
            identify_precise_features(experiments, include=("bogus",))

    def test_empty_experiments_raises(self) -> None:
        with pytest.raises(HABITAPIError, match="at least one"):
            identify_precise_features({})


class TestPreciseFeatureSet:
    def test_round_trip(self, tmp_path) -> None:
        experiments = {
            "repeatability": pd.DataFrame(
                {"value": 0.9, "lcl": [0.6, 0.4], "ucl": 0.95, "n_voxels": 100},
                index=pd.Index(["f1", "f2"], name="feature"),
            )
        }
        result = identify_precise_features(experiments, lcl_threshold=0.5)
        path = result.save(tmp_path / "precise.json")
        loaded = PreciseFeatureSet.load(path)
        assert loaded.feature_names == result.feature_names
        assert loaded.lcl_threshold == result.lcl_threshold
        assert loaded.experiments == result.experiments
        pd.testing.assert_frame_equal(
            loaded.panels["repeatability"], result.panels["repeatability"]
        )

    def test_to_frame_marks_precise(self) -> None:
        experiments = {
            "repeatability": pd.DataFrame(
                {"value": 0.9, "lcl": [0.6, 0.4], "ucl": 0.95, "n_voxels": 100},
                index=pd.Index(["f1", "f2"], name="feature"),
            )
        }
        result = identify_precise_features(experiments, lcl_threshold=0.5)
        evidence = result.to_frame()
        assert set(evidence.columns) >= {
            "experiment",
            "feature",
            "lcl",
            "precise",
        }
        assert evidence.set_index("feature")["precise"].to_dict() == {
            "f1": True,
            "f2": False,
        }

    def test_load_rejects_foreign_file(self, tmp_path) -> None:
        path = tmp_path / "other.json"
        path.write_text('{"format": "something.else"}', encoding="utf-8")
        with pytest.raises(HABITAPIError, match="Not a PreciseFeatureSet"):
            PreciseFeatureSet.load(path)

    def test_preprocessor_restricts_a_frame_to_the_precise_set(self) -> None:
        """The bridge to habitat computation: cluster exactly these features."""
        experiments = {
            "repeatability": pd.DataFrame(
                {"value": 0.9, "lcl": [0.6, 0.4], "ucl": 0.95, "n_voxels": 100},
                index=pd.Index(["f1", "f2"], name="feature"),
            )
        }
        result = identify_precise_features(experiments, lcl_threshold=0.5)
        method = result.preprocessor()
        block = pd.DataFrame(
            {"f1": [1.0, 2.0], "f2": [3.0, 4.0], "f3": [5.0, 6.0]}
        )
        out = method.transform(block, method.fit(block))
        assert list(out.columns) == ["f1"]
        assert method.spec.name == "feature_whitelist"


class TestHabitatStability:
    def test_identical_maps_score_one(self) -> None:
        reference = make_habitat_map("P1")
        frame = habitat_stability(reference, [make_habitat_map("P1")])
        assert set(frame["dice"]) == {1.0}
        assert sorted(frame["habitat_id"]) == [1, 2]

    def test_swapped_labels_match_via_hungarian(self) -> None:
        reference = make_habitat_map("P1")
        swapped_array = np.asarray(reference.label_array).copy()
        swapped_array[swapped_array == 1] = 9
        swapped_array[swapped_array == 2] = 1
        swapped_array[swapped_array == 9] = 2
        swapped = HabitatMap(
            subject_id="P1",
            label_array=swapped_array,
            geometry=reference.geometry,
            model_id="other-model",
            habitat_ids=(1, 2),
            provenance=provenance(),
        )
        frame = habitat_stability(reference, [swapped])
        assert set(frame["dice"]) == {1.0}
        matched = frame.set_index("habitat_id")["matched_id"].to_dict()
        assert matched == {1: 2, 2: 1}

    def test_partial_overlap(self) -> None:
        reference = make_habitat_map("P1")
        moved_array = np.zeros((4, 4, 4), dtype=np.int32)
        moved_array[1:3, 0:2, 0:2] = 1  # habitat 1 shifted by one voxel in z
        moved_array[2:4, 0:2, 0:2] = 2
        moved = HabitatMap(
            subject_id="P1",
            label_array=moved_array,
            geometry=reference.geometry,
            model_id="other-model",
            habitat_ids=(1, 2),
            provenance=provenance(),
        )
        frame = habitat_stability(reference, [moved])
        dice_h1 = frame.loc[frame["habitat_id"] == 1, "dice"].iloc[0]
        # moved[1:3]=1 is overwritten by moved[2:4]=2 at z=2, so habitat 1
        # keeps only the z=1 plane: intersection 4 voxels, sizes 8 and 4.
        assert dice_h1 == pytest.approx(2.0 * 4.0 / (8.0 + 4.0))

    def test_unmatched_habitat_scores_zero(self) -> None:
        reference = make_habitat_map("P1")
        single_array = np.zeros((4, 4, 4), dtype=np.int32)
        single_array[0:2, 0:2, 0:2] = 1
        single = HabitatMap(
            subject_id="P1",
            label_array=single_array,
            geometry=reference.geometry,
            model_id="other-model",
            habitat_ids=(1,),
            provenance=provenance(),
        )
        frame = habitat_stability(reference, [single])
        assert frame.loc[frame["habitat_id"] == 2, "dice"].iloc[0] == 0.0
        assert pd.isna(frame.loc[frame["habitat_id"] == 2, "matched_id"].iloc[0])

    def test_shape_mismatch_raises(self) -> None:
        reference = make_habitat_map("P1")
        other = HabitatMap(
            subject_id="P1",
            label_array=np.zeros((5, 5, 5), dtype=np.int32),
            geometry=Geometry.from_array((5, 5, 5)),
            model_id="other-model",
            habitat_ids=(1,),
            provenance=provenance(),
        )
        with pytest.raises(HABITAPIError, match="shape"):
            habitat_stability(reference, [other])

    def test_empty_raises(self) -> None:
        with pytest.raises(HABITAPIError, match="at least one"):
            habitat_stability(make_habitat_map("P1"), [])

    def test_centroid_mean_intensity_recovers_swapped_ids(self) -> None:
        """Mean-intensity Hungarian pairing, then ordinary Dice on that pair."""
        reference = make_habitat_map("P1")
        swapped_array = np.asarray(reference.label_array).copy()
        swapped_array[swapped_array == 1] = 9
        swapped_array[swapped_array == 2] = 1
        swapped_array[swapped_array == 9] = 2
        swapped = HabitatMap(
            subject_id="P1",
            label_array=swapped_array,
            geometry=reference.geometry,
            model_id="other-model",
            habitat_ids=(1, 2),
            provenance=provenance(),
        )
        image = np.zeros(reference.label_array.shape, dtype=np.float64)
        image[reference.label_array == 1] = 1.0
        image[reference.label_array == 2] = 10.0
        frame = habitat_stability(
            reference, [swapped], method="centroid", image=image
        )
        assert set(frame["dice"]) == {1.0}
        matched = frame.set_index("habitat_id")["matched_id"].to_dict()
        assert matched == {1: 2, 2: 1}

    def test_unknown_method_raises(self) -> None:
        reference = make_habitat_map("P1")
        with pytest.raises(HABITAPIError, match="method"):
            habitat_stability(reference, [reference], method="dice")  # type: ignore[arg-type]


class TestAlignHabitatMap:
    def test_permuted_labels_become_comparable(self) -> None:
        """Independent clustering with swapped ids remaps onto the reference."""
        reference = make_habitat_map("P1")
        swapped_array = np.asarray(reference.label_array).copy()
        swapped_array[swapped_array == 1] = 9
        swapped_array[swapped_array == 2] = 1
        swapped_array[swapped_array == 9] = 2
        moving = HabitatMap(
            subject_id="P1",
            label_array=swapped_array,
            geometry=reference.geometry,
            model_id="other-model",
            habitat_ids=(1, 2),
            provenance=provenance(),
        )
        image = np.zeros(reference.label_array.shape, dtype=np.float64)
        image[reference.label_array == 1] = 1.0
        image[reference.label_array == 2] = 10.0
        aligned = align_habitat_map(reference, moving, image=image)
        assert np.array_equal(aligned.label_array, reference.label_array)
        assert aligned.model_id == reference.model_id
        # After remap, raw pixel disagreement is spatial, not an id swap.
        disagree = (
            (aligned.label_array != reference.label_array)
            & ((aligned.label_array > 0) | (reference.label_array > 0))
        )
        assert int(np.count_nonzero(disagree)) == 0

    def test_explicit_centroids_recover_permutation(self) -> None:
        """Cluster centres from two fits drive the test-retest assignment."""
        reference = make_habitat_map("P1")
        swapped_array = np.asarray(reference.label_array).copy()
        swapped_array[swapped_array == 1] = 9
        swapped_array[swapped_array == 2] = 1
        swapped_array[swapped_array == 9] = 2
        moving = HabitatMap(
            subject_id="P1",
            label_array=swapped_array,
            geometry=reference.geometry,
            model_id="other-model",
            habitat_ids=(1, 2),
            provenance=provenance(),
        )
        aligned = align_habitat_map(
            reference,
            moving,
            reference_centroids=np.array([[0.0], [10.0]]),
            moving_centroids=np.array([[10.0], [0.0]]),
        )
        assert np.array_equal(aligned.label_array, reference.label_array)

    def test_same_model_id_is_identity(self) -> None:
        """Apply-same-model maps keep their labels even if ids look swapped."""
        reference = make_habitat_map("P1")
        swapped_array = np.asarray(reference.label_array).copy()
        swapped_array[swapped_array == 1] = 9
        swapped_array[swapped_array == 2] = 1
        swapped_array[swapped_array == 9] = 2
        moving = HabitatMap(
            subject_id="P1",
            label_array=swapped_array,
            geometry=reference.geometry,
            model_id=reference.model_id,
            habitat_ids=(1, 2),
            provenance=provenance(),
        )
        image = np.zeros(reference.label_array.shape, dtype=np.float64)
        image[reference.label_array == 1] = 1.0
        image[reference.label_array == 2] = 10.0
        aligned = align_habitat_map(reference, moving, image=image)
        assert aligned is moving
        assert np.array_equal(aligned.label_array, swapped_array)

    def test_force_aligns_shared_model(self) -> None:
        """force=True remaps even when model_id already matches."""
        reference = make_habitat_map("P1")
        swapped_array = np.asarray(reference.label_array).copy()
        swapped_array[swapped_array == 1] = 9
        swapped_array[swapped_array == 2] = 1
        swapped_array[swapped_array == 9] = 2
        moving = HabitatMap(
            subject_id="P1",
            label_array=swapped_array,
            geometry=reference.geometry,
            model_id=reference.model_id,
            habitat_ids=(1, 2),
            provenance=provenance(),
        )
        aligned = align_habitat_map(
            reference,
            moving,
            method="overlap",
            force=True,
        )
        assert np.array_equal(aligned.label_array, reference.label_array)

    def test_overlap_2_3_swap_on_moving_only(self) -> None:
        """Overlap remaps a 2<->3 permutation on moving; disagreement ignores it.

        Independent one_step fits of the same spec on the same subject share
        a model_id (subject-id digest, not image content). force=True is
        what actually applies the remap in that situation.
        """
        labels = np.zeros((6, 4, 4), dtype=np.int32)
        labels[0:2, 0:2, 0:2] = 1
        labels[2:4, 0:2, 0:2] = 2
        labels[4:6, 0:2, 0:2] = 3
        swapped = labels.copy()
        swapped[labels == 2] = 9
        swapped[labels == 3] = 2
        swapped[swapped == 9] = 3
        reference = HabitatMap(
            subject_id="P1",
            label_array=labels,
            geometry=Geometry.from_array((6, 4, 4)),
            model_id="shared-one-step-id",
            habitat_ids=(1, 2, 3),
            provenance=provenance(),
        )
        moving = HabitatMap(
            subject_id="P1",
            label_array=swapped,
            geometry=reference.geometry,
            model_id="shared-one-step-id",
            habitat_ids=(1, 2, 3),
            provenance=provenance(),
        )
        skipped = align_habitat_map(reference, moving, method="overlap")
        assert skipped is moving
        aligned = align_habitat_map(
            reference, moving, method="overlap", force=True
        )
        assert np.array_equal(reference.label_array, labels)
        assert np.array_equal(aligned.label_array, labels)
        assert not np.array_equal(moving.label_array, labels)
        raw_disagree = (
            (moving.label_array != reference.label_array)
            & ((moving.label_array > 0) | (reference.label_array > 0))
        )
        aligned_disagree = (
            (aligned.label_array != reference.label_array)
            & ((aligned.label_array > 0) | (reference.label_array > 0))
        )
        assert int(np.count_nonzero(raw_disagree)) > 0
        assert int(np.count_nonzero(aligned_disagree)) == 0

    def test_spatial_disagreement_survives_remap(self) -> None:
        """A shifted habitat still disagrees after ids are aligned."""
        reference = make_habitat_map("P1")
        moved_array = np.zeros((4, 4, 4), dtype=np.int32)
        moved_array[1:3, 0:2, 0:2] = 2  # swapped id AND shifted
        moved_array[2:4, 0:2, 0:2] = 1
        moving = HabitatMap(
            subject_id="P1",
            label_array=moved_array,
            geometry=reference.geometry,
            model_id="other-model",
            habitat_ids=(1, 2),
            provenance=provenance(),
        )
        aligned = align_habitat_map(reference, moving, method="overlap")
        disagree = (
            (aligned.label_array != reference.label_array)
            & ((aligned.label_array > 0) | (reference.label_array > 0))
        )
        assert int(np.count_nonzero(disagree)) > 0
