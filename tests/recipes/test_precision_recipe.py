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

"""End-to-end discrimination test for the precise-feature recipe.

The synthetic cohort below is built so that the GROUND TRUTH about every
feature's precision is known by construction:

* the phantom's band contrast is comparable to its noise level, so raw
  voxel intensities are unreliable while spatial averages are stable;
* the toy extractor emits four columns whose sensitivities are engineered
  one dial at a time (see the extractor's docstring).

Running the full chain -- perturb, extract, ICC, aggregate, select -- must
recover exactly the one column stable under every experiment. Each
experiment's evidence panel is additionally checked to veto exactly the
columns it was designed to veto, proving the selection fails features for
the right reason rather than by accident.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import uniform_filter

from habit.contracts import (
    ArrayImageRef,
    Cohort,
    Geometry,
    Subject,
    VoxelFeatureField,
)
from habit.voxel_features import (
    aligned_image,
    build_voxel_field,
    roi_voxels,
)
from habit.recipes import identify_precise_voxel_features
from habit.recipes.precision import prior2024_voxel_extract_params
from habit.spec.specs import Spec

_BASE_RADIUS = 3
_BASE_WIDTH = 12.0
# Noise above the 50-unit band contrast: raw intensities carry an SNR
# below 1, while a 5^3 average divides the noise by ~11.
_NOISE_SIGMA = 80.0
_SMOOTH_SIZE = 5
_SPECKLE_SIZE = 3


def _local_std(array: np.ndarray, size: int) -> np.ndarray:
    """Local standard deviation over a ``size``^3 uniform window."""
    mean = uniform_filter(array, size=size)
    mean_sq = uniform_filter(array * array, size=size)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def _phantom_subject(subject_id: str, seed: int) -> Subject:
    """
    Banded phantom with realistic noise and a block ROI.

    Three intensity bands (150/200/250 on a 100 background) provide the
    between-voxel signal; iid Gaussian noise at ``_NOISE_SIGMA`` makes the
    raw intensity noise-dominated. The ROI covers the bands with margin so
    the perturbation's sub-voxel shift and rotation never move it across
    the array boundary.
    """
    shape = (12, 40, 40)
    geometry = Geometry.from_array(shape)
    rng = np.random.default_rng(seed)
    array = np.full(shape, 100.0, dtype=np.float64)
    array[2:5, 8:32, 8:32] = 150.0
    array[5:8, 8:32, 8:32] = 200.0
    array[8:11, 8:32, 8:32] = 250.0
    array += rng.normal(0.0, _NOISE_SIGMA, size=shape)
    mask = np.zeros(shape, dtype=np.int32)
    mask[2:11, 8:32, 8:32] = 1
    return Subject(
        subject_id=subject_id,
        images={"CT": ArrayImageRef(array=array, geometry=geometry)},
        masks={"tumor": ArrayImageRef(array=mask, geometry=geometry)},
    )


def _phantom_cohort(n_subjects: int) -> Cohort:
    """Cohort of phantoms sharing the band layout but not the noise draw."""
    return Cohort(
        [_phantom_subject(f"P{i}", seed=100 + i) for i in range(n_subjects)],
        name="phantoms",
    )


class _SensitivityToyExtractor:
    """
    Voxel extractor emitting four columns with engineered sensitivities.

    ``local_mean``
        5^3 local average. Averaging 125 voxels suppresses the phantom
        noise, so the column survives every experiment -- the precise
        feature. Models stable first-order features.
    ``raw_voxel``
        The raw intensity. The phantom noise is comparable to the band
        contrast and the perturbation chain adds a fresh noise draw, so
        the column fails REPEATABILITY; being setting-invariant it passes
        both reproducibility experiments. Models noise-dominated features.
    ``radius_mix``
        ``local_mean`` at the base kernel radius, ``speckle`` (3^3 local
        std, nearly uncorrelated with the smooth mean) at any other
        radius. Passes repeatability -- it IS the stable column at the
        base setting -- but the two settings produce unrelated maps, so it
        fails REPRODUCIBILITY_KERNEL_RADIUS. Models features whose
        response surface changes with the neighbourhood size.
    ``width_mix``
        The same dial on the bin width; fails
        REPRODUCIBILITY_BIN_WIDTH. Models features whose voxel ranking
        changes with discretisation.
    """

    def __init__(self, kernel_radius: int, bin_width: float) -> None:
        self._kernel_radius = kernel_radius
        self._bin_width = bin_width

    @property
    def spec(self) -> Spec:
        """The setting pair this instance extracts at."""
        return Spec(
            name="sensitivity_toy",
            params={
                "kernel_radius": self._kernel_radius,
                "bin_width": self._bin_width,
            },
        )

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """Extract the four engineered columns on the subject's ROI."""
        mask, inside, voxel_index = roi_voxels(subject, None)
        image = aligned_image(subject, "CT", mask, owner="sensitivity_toy")
        image = image.astype(np.float64)
        local_mean = uniform_filter(image, size=_SMOOTH_SIZE)
        speckle = _local_std(image, size=_SPECKLE_SIZE)
        radius_mix = local_mean if self._kernel_radius == _BASE_RADIUS else speckle
        width_mix = local_mean if self._bin_width == _BASE_WIDTH else speckle
        columns = np.column_stack(
            [
                local_mean[inside],
                image[inside],
                radius_mix[inside],
                width_mix[inside],
            ]
        )
        return build_voxel_field(
            subject,
            mask,
            voxel_index,
            ("local_mean", "raw_voxel", "radius_mix", "width_mix"),
            columns,
            self.spec,
        )


def _factory(kernel_radius: int, bin_width: float) -> _SensitivityToyExtractor:
    """Extractor factory of the signature the recipe expects."""
    return _SensitivityToyExtractor(kernel_radius=kernel_radius, bin_width=bin_width)


def _run(cohort: Cohort, seed: int = 11):
    """Run the full recipe on the phantom cohort with both experiments on."""
    return identify_precise_voxel_features(
        cohort,
        extractor_factory=_factory,
        kernel_radii=(1, _BASE_RADIUS),
        bin_widths=(25.0, _BASE_WIDTH),
        base_kernel_radius=_BASE_RADIUS,
        base_bin_width=_BASE_WIDTH,
        seed=seed,
        show_progress=False,
    )


@pytest.mark.unit
def test_recipe_selects_exactly_the_stable_feature() -> None:
    """Only the column stable under every experiment may be selected."""
    result = _run(_phantom_cohort(3))
    assert result.feature_names == ("local_mean",)


@pytest.mark.unit
def test_each_experiment_vetoes_exactly_its_sensitive_column() -> None:
    """Every panel rejects the column engineered to fail it, and only those."""
    result = _run(_phantom_cohort(3))

    repeat = result.panels["repeatability"]
    assert repeat.loc["local_mean", "lcl"] >= 0.5
    assert repeat.loc["raw_voxel", "lcl"] < 0.5
    # The mix columns equal local_mean at the base setting, so they pass here.
    assert repeat.loc["radius_mix", "lcl"] >= 0.5
    assert repeat.loc["width_mix", "lcl"] >= 0.5

    radius = result.panels["reproducibility_kernel_radius"]
    assert radius.loc["radius_mix", "lcl"] < 0.5
    assert radius.loc["local_mean", "lcl"] >= 0.5
    # Setting-invariant columns are unaffected by the radius sweep.
    assert radius.loc["raw_voxel", "lcl"] >= 0.5
    assert radius.loc["width_mix", "lcl"] >= 0.5

    width = result.panels["reproducibility_bin_width"]
    assert width.loc["width_mix", "lcl"] < 0.5
    assert width.loc["local_mean", "lcl"] >= 0.5
    assert width.loc["raw_voxel", "lcl"] >= 0.5
    assert width.loc["radius_mix", "lcl"] >= 0.5


@pytest.mark.unit
def test_recipe_is_deterministic_for_a_fixed_seed() -> None:
    """The same seed reproduces the selection and every panel value."""
    first = _run(_phantom_cohort(2), seed=5)
    second = _run(_phantom_cohort(2), seed=5)
    assert first.feature_names == second.feature_names
    assert set(first.panels) == set(second.panels)
    for name in first.panels:
        pd.testing.assert_frame_equal(first.panels[name], second.panels[name])


# ---------------------------------------------------------------------------
# Graded sensitivity: a feature family with a continuous noise dial
# ---------------------------------------------------------------------------

#: Noise-fraction dial values, from fully stable to noise-dominated.
_GRADIENT_LAMBDAS: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5)


class _GradientToyExtractor:
    """
    Feature family ``local_mean + lambda * (image - local_mean)``.

    The ``local_mean`` component is stable under the perturbation chain;
    the residual ``image - local_mean`` is the noise-dominated part.
    Lambda therefore dials the noise fraction CONTINUOUSLY -- unlike the
    binary _SensitivityToyExtractor switches -- so the measured
    repeatability ICC must decrease monotonically as lambda grows, and the
    screen's threshold must cut the family at a single point. The columns
    are setting-invariant, so only the repeatability experiment decides.
    """

    def __init__(self, kernel_radius: int, bin_width: float) -> None:
        # Settings are accepted to satisfy the factory signature; the
        # family is deliberately setting-invariant.
        self._spec = Spec(
            name="gradient_toy",
            params={"kernel_radius": kernel_radius, "bin_width": bin_width},
        )

    @property
    def spec(self) -> Spec:
        """The setting pair this instance extracts at."""
        return self._spec

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """Extract one column per lambda, in ascending noise fraction."""
        mask, inside, voxel_index = roi_voxels(subject, None)
        image = aligned_image(subject, "CT", mask, owner="gradient_toy")
        image = image.astype(np.float64)
        local_mean = uniform_filter(image, size=_SMOOTH_SIZE)
        residual = image - local_mean
        columns = np.column_stack(
            [(local_mean + lam * residual)[inside] for lam in _GRADIENT_LAMBDAS]
        )
        names = tuple(f"lam_{lam}" for lam in _GRADIENT_LAMBDAS)
        return build_voxel_field(subject, mask, voxel_index, names, columns, self.spec)


def _gradient_run(cohort: Cohort, seed: int = 11):
    """Screen the gradient family; only the repeatability experiment runs."""
    return identify_precise_voxel_features(
        cohort,
        extractor_factory=lambda radius, width: _GradientToyExtractor(radius, width),
        kernel_radii=(_BASE_RADIUS,),
        bin_widths=(_BASE_WIDTH,),
        seed=seed,
        show_progress=False,
    )


@pytest.mark.unit
def test_repeatability_icc_decreases_monotonically_with_noise_fraction() -> None:
    """The measured ICC scale is calibrated: more noise -> lower ICC."""
    result = _gradient_run(_phantom_cohort(3))
    panel = result.panels["repeatability"]
    names = [f"lam_{lam}" for lam in _GRADIENT_LAMBDAS]
    for column in ("value", "lcl"):
        series = [panel.loc[name, column] for name in names]
        for higher_signal, lower_signal in zip(series, series[1:]):
            assert higher_signal >= lower_signal - 1e-6


@pytest.mark.unit
def test_selection_boundary_is_a_prefix_of_the_noise_gradient() -> None:
    """
    The threshold cuts the family at a single point of the noise dial.

    The fully stable column and the mildly noisy one must be selected, the
    noise-dominated ones must not, and the selected set must be a PREFIX of
    the ascending lambda order -- a non-contiguous selection would mean the
    screen responds to something other than the noise fraction.
    """
    result = _gradient_run(_phantom_cohort(3))
    selected = set(result.feature_names)
    assert "lam_0.0" in selected
    assert "lam_0.2" in selected
    assert "lam_1.0" not in selected
    assert "lam_1.5" not in selected
    ordered = [f"lam_{lam}" for lam in _GRADIENT_LAMBDAS]
    boundary = sum(name in selected for name in ordered)
    assert [name in selected for name in ordered] == [True] * boundary + [
        False
    ] * (len(ordered) - boundary)


@pytest.mark.unit
def test_prior2024_extract_params_force_voxel_array_shift_zero() -> None:
    """Prior YAML omits voxelArrayShift; the overlay must write 0."""
    merged = prior2024_voxel_extract_params(
        {"setting": {"voxelArrayShift": 300, "binWidth": 25}},
        bin_width=12.0,
    )
    assert merged["setting"]["voxelArrayShift"] == 0
    assert merged["setting"]["binWidth"] == 12.0
    assert merged["setting"]["interpolator"] == "sitkBSpline"
    assert merged["voxelSetting"]["initValue"] != merged["voxelSetting"]["initValue"]
