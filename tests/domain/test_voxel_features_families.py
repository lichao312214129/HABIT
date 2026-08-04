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
"""Tests for the local-entropy, kinetic and concat voxel feature families.

``voxel_radiomics`` is exercised through the shared kernel
(:mod:`tests.kernels.test_voxel_maps`) plus the golden gate rather than here,
because a per-voxel PyRadiomics pass is too slow for the unit tier.
"""

from __future__ import annotations

import numpy as np
import pytest

from habit.contracts import VoxelFeatureField
from habit.domain.protocols import VoxelFeatureExtractor
from habit.domain.voxel_features import (
    ConcatVoxelFeatures,
    KineticVoxelFeatures,
    LocalEntropyVoxelFeatures,
    RawVoxelFeatures,
    VoxelFeatureExtractorRegistry,
    VoxelRadiomicsFeatures,
)
from habit.exceptions import GeometryError, HABITAPIError
from habit.kernels.voxel_texture import local_entropy_map

from .conftest import make_subject

#: Acquisition times for the synthetic dynamic series, 30 s apart.
_TIMES = {"LAP": "10-00-30", "PVP": "10-01-00", "delay_3min": "10-03-00"}


def _dynamic_subject(subject_id: str = "P1"):
    """Build a subject carrying the four v0.1 contrast phases."""
    return make_subject(
        subject_id,
        modalities=("pre_contrast", "LAP", "PVP", "delay_3min"),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "extractor",
    [
        LocalEntropyVoxelFeatures(modalities=["T1"]),
        VoxelRadiomicsFeatures(modalities=["T1"]),
        KineticVoxelFeatures(timestamps={"P1": _TIMES}),
        ConcatVoxelFeatures(
            extractors=[
                {"name": "raw", "params": {"modalities": ["T1"]}},
                {"name": "local_entropy", "params": {"modalities": ["T1"]}},
            ]
        ),
    ],
    ids=["local_entropy", "voxel_radiomics", "kinetic", "concat"],
)
def test_new_families_satisfy_the_protocol(extractor: object) -> None:
    """Each new family structurally satisfies the domain protocol."""
    assert isinstance(extractor, VoxelFeatureExtractor)


@pytest.mark.unit
def test_every_v0_voxel_family_is_registered() -> None:
    """The v0.1 voxel families are all constructible by name."""
    available = set(VoxelFeatureExtractorRegistry.available())
    assert {"raw", "voxel_radiomics", "kinetic", "local_entropy", "concat"} <= available


@pytest.mark.unit
def test_local_entropy_matches_the_shared_kernel() -> None:
    """
    The extractor is the kernel restricted to the ROI, nothing more.

    Computing entropy on the whole image before masking is deliberate: ROI
    border voxels must still see their true neighbourhood.
    """
    subject = make_subject("P1")
    field = LocalEntropyVoxelFeatures(modalities=["T1"], kernel_size=3, bins=8)(subject)

    assert isinstance(field, VoxelFeatureField)
    assert field.feature_names == ("local_entropy-T1",)
    assert field.provenance.produced_by == "voxel_feature_extractor.local_entropy"

    inside = np.asarray(subject.mask("tumor").data) > 0
    expected = local_entropy_map(
        np.asarray(subject.image("T1").data), kernel_size=3, bins=8
    )[inside]
    np.testing.assert_allclose(field.values[:, 0], expected)


@pytest.mark.unit
def test_local_entropy_columns_follow_modality_order() -> None:
    """One column per modality, suffixed the v0.1 way."""
    subject = make_subject("P1", modalities=("T1", "T2"))
    field = LocalEntropyVoxelFeatures(modalities=["T2", "T1"])(subject)
    assert field.feature_names == ("local_entropy-T2", "local_entropy-T1")


@pytest.mark.unit
def test_local_entropy_even_kernel_is_made_odd() -> None:
    """An even neighbourhood cannot be centred, so it is incremented."""
    subject = make_subject("P1")
    even = LocalEntropyVoxelFeatures(modalities=["T1"], kernel_size=4)(subject)
    odd = LocalEntropyVoxelFeatures(modalities=["T1"], kernel_size=5)(subject)
    np.testing.assert_allclose(even.values, odd.values)


@pytest.mark.unit
def test_local_entropy_is_zero_on_a_constant_image() -> None:
    """A homogeneous neighbourhood carries no information."""
    constant = np.full((5, 5, 5), 3.0)
    entropy = local_entropy_map(constant, kernel_size=3, bins=8)
    # Only the border sees implicit zeros outside the array; the interior is
    # fully homogeneous and therefore exactly zero-entropy.
    assert entropy[2, 2, 2] == pytest.approx(0.0)


@pytest.mark.unit
def test_kinetic_slopes_use_each_subjects_acquisition_times() -> None:
    """Slopes are enhancement per second over the true phase intervals."""
    subject = _dynamic_subject()
    field = KineticVoxelFeatures(timestamps={"P1": _TIMES})(subject)

    assert field.feature_names == (
        "wash_in_slope",
        "wash_out_slope_lap_pvp",
        "wash_out_slope_pvp_dp",
    )
    inside = np.asarray(subject.mask("tumor").data) > 0
    phase = {
        name: np.asarray(subject.image(name).data)[inside]
        for name in ("pre_contrast", "LAP", "PVP", "delay_3min")
    }
    # pre_contrast is placed 25 s before LAP; the recorded gaps are 30 s and 120 s.
    np.testing.assert_allclose(
        field.values[:, 0],
        np.clip(phase["LAP"] - phase["pre_contrast"], 0.0, None) / (25.0 + 1e-6),
    )
    np.testing.assert_allclose(
        field.values[:, 1], (phase["PVP"] - phase["LAP"]) / (30.0 + 1e-6)
    )
    np.testing.assert_allclose(
        field.values[:, 2], (phase["delay_3min"] - phase["PVP"]) / (120.0 + 1e-6)
    )


@pytest.mark.unit
def test_kinetic_clips_negative_wash_in() -> None:
    """A drop after contrast is no enhancement, not negative wash-in."""
    subject = _dynamic_subject()
    field = KineticVoxelFeatures(timestamps={"P1": _TIMES})(subject)
    assert np.all(field.values[:, 0] >= 0.0)


@pytest.mark.unit
def test_kinetic_reports_missing_inputs_precisely() -> None:
    """Absent phase images and absent acquisition times fail differently."""
    subject = make_subject("P1", modalities=("pre_contrast", "LAP"))
    with pytest.raises(HABITAPIError, match="phase images"):
        KineticVoxelFeatures(timestamps={"P1": _TIMES})(subject)

    complete = _dynamic_subject()
    with pytest.raises(HABITAPIError, match="acquisition times"):
        KineticVoxelFeatures(timestamps={"other": _TIMES})(complete)

    with pytest.raises(HABITAPIError, match="acquisition times"):
        KineticVoxelFeatures(timestamps={"P1": {"LAP": "10-00-30"}})(complete)


@pytest.mark.unit
def test_kinetic_requires_four_phases() -> None:
    """The four phases have fixed roles, so the count is not negotiable."""
    with pytest.raises(HABITAPIError, match="exactly four phases"):
        KineticVoxelFeatures(timestamps={}, phases=("pre_contrast", "LAP"))


@pytest.mark.unit
def test_concat_joins_children_row_for_row() -> None:
    """Children describe the same ROI voxels, so columns simply stack."""
    subject = make_subject("P1", modalities=("T1", "T2"))
    concat = ConcatVoxelFeatures(
        extractors=[
            {"name": "raw", "params": {"modalities": ["T1"]}},
            {"name": "local_entropy", "params": {"modalities": ["T2"]}},
        ]
    )
    field = concat(subject)

    assert field.feature_names == ("T1", "local_entropy-T2")
    raw_only = RawVoxelFeatures(modalities=["T1"])(subject)
    entropy_only = LocalEntropyVoxelFeatures(modalities=["T2"])(subject)
    np.testing.assert_allclose(field.values[:, 0], raw_only.values[:, 0])
    np.testing.assert_allclose(field.values[:, 1], entropy_only.values[:, 0])
    np.testing.assert_array_equal(field.voxel_index, raw_only.voxel_index)
    assert field.provenance.produced_by == "voxel_feature_extractor.concat"


@pytest.mark.unit
def test_concat_rejects_degenerate_and_ambiguous_compositions() -> None:
    """One child is pointless, and duplicate columns would be ambiguous."""
    with pytest.raises(HABITAPIError, match="at least two"):
        ConcatVoxelFeatures(extractors=[{"name": "raw", "params": {"modalities": ["T1"]}}])
    with pytest.raises(HABITAPIError, match="missing 'name'"):
        ConcatVoxelFeatures(extractors=[{"params": {}}, {"params": {}}])

    subject = make_subject("P1")
    same_twice = ConcatVoxelFeatures(
        extractors=[
            {"name": "raw", "params": {"modalities": ["T1"]}},
            {"name": "raw", "params": {"modalities": ["T1"]}},
        ]
    )
    with pytest.raises(HABITAPIError, match="repeats column"):
        same_twice(subject)


@pytest.mark.unit
def test_new_families_round_trip_through_their_specs() -> None:
    """Each family's spec rebuilds an equivalent extractor via the registry."""
    for extractor in (
        LocalEntropyVoxelFeatures(modalities=["T1"], kernel_size=5, bins=16),
        VoxelRadiomicsFeatures(modalities=["T1"], kernel_radius=2),
        KineticVoxelFeatures(timestamps={"P1": _TIMES}),
        ConcatVoxelFeatures(
            extractors=[
                {"name": "raw", "params": {"modalities": ["T1"]}},
                {"name": "local_entropy", "params": {"modalities": ["T1"]}},
            ]
        ),
    ):
        spec = extractor.spec
        rebuilt = VoxelFeatureExtractorRegistry.create(spec.name, **spec.params)
        assert rebuilt.spec.to_dict() == spec.to_dict()


@pytest.mark.unit
def test_voxel_radiomics_rejects_conflicting_settings() -> None:
    """Settings come from a file or a mapping, never both."""
    with pytest.raises(HABITAPIError, match="mutually exclusive"):
        VoxelRadiomicsFeatures(params_file="a.yaml", params={"binWidth": 25})
    with pytest.raises(HABITAPIError, match="kernel_radius"):
        VoxelRadiomicsFeatures(kernel_radius=0)


@pytest.mark.unit
def test_families_reject_a_modality_off_the_roi_grid() -> None:
    """A modality on a different grid cannot be paired with the ROI."""
    from habit.contracts import Subject

    subject = make_subject("P1")
    other = make_subject("P1", shape=(4, 4, 4))
    mixed = Subject(
        subject_id=subject.subject_id,
        images={"T1": other.images["T1"]},
        masks=dict(subject.masks),
    )
    with pytest.raises(GeometryError):
        LocalEntropyVoxelFeatures(modalities=["T1"])(mixed)
