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
"""End-to-end tests for SubjectPipeline on fully synthetic arrays."""

from __future__ import annotations

import numpy as np
import pytest

from habit.api.exceptions import HABITAPIError
from habit.contracts import Cohort, HabitatMap
from habit.feature_preprocessing import CohortPreprocessingChain, SubjectPreprocessingChain, build_methods
from habit.habitat_features import HabitatVolumeFeatures, IthHabitatFeatures, MsiHabitatFeatures
from habit.habitat_model import KMeansHabitatModelFitter
from habit.voxel_features import RawVoxelFeatures
from habit.supervoxel import SlicSupervoxelizer
from habit.pipeline import SubjectPipeline
from habit.pipeline import voxel_units

from .conftest import make_field, make_subject


def _fitted_pipeline(*, supervoxels: bool = True, seed: int = 11) -> SubjectPipeline:
    """Fit a two-habitat model on synthetic subjects and build the pipeline."""
    voxel_features = RawVoxelFeatures(modalities=["T1"])
    supervoxelizer = SlicSupervoxelizer(n_supervoxels=8) if supervoxels else None
    cohort = Cohort([make_subject(f"S{i}", seed=i) for i in range(3)])
    if supervoxelizer is None:
        units = [voxel_units(voxel_features(subject)) for subject in cohort]
    else:
        units = [supervoxelizer(voxel_features(subject)) for subject in cohort]
    fitter = KMeansHabitatModelFitter(n_habitats=2, n_init=5)
    fitter.set_random_state(seed)
    model = fitter.fit(units, cohort=cohort)
    return SubjectPipeline(voxel_features, supervoxelizer, model.assigner())


@pytest.mark.unit
def test_pipeline_labels_unseen_subject() -> None:
    """The composed chain labels a subject end to end."""
    pipeline = _fitted_pipeline()
    habitat_map = pipeline(make_subject("new", seed=42))
    assert isinstance(habitat_map, HabitatMap)
    assert habitat_map.subject_id == "new"
    assert habitat_map.model_id == pipeline.habitat_assigner.model.model_id
    labels = set(np.unique(np.asarray(habitat_map.label_array)))
    assert labels <= {0, 1, 2}
    assert labels - {0}  # at least one habitat present


@pytest.mark.unit
def test_pipeline_assign_reuses_precomputed_units() -> None:
    """``assign`` / ``label_and_describe`` match ``__call__`` without re-units."""
    pipeline = _fitted_pipeline()
    subject = make_subject("new", seed=17)
    units = pipeline.units(subject)
    from_call = pipeline(subject)
    from_units, prepared = pipeline.assign(units)
    map_described, table, prepared2 = pipeline.label_and_describe(
        subject, units, [HabitatVolumeFeatures()]
    )
    np.testing.assert_array_equal(
        np.asarray(from_call.label_array), np.asarray(from_units.label_array)
    )
    np.testing.assert_array_equal(
        np.asarray(from_units.label_array), np.asarray(map_described.label_array)
    )
    assert prepared.subject_id == prepared2.subject_id == subject.subject_id
    assert table is not None
    assert subject.subject_id in set(table.frame["subject"].astype(str))


@pytest.mark.unit
def test_pipeline_is_deterministic_for_fixed_seed() -> None:
    """A fitted pipeline assigns identical labels on repeated calls."""
    pipeline = _fitted_pipeline()
    subject = make_subject("new", seed=5)
    first = pipeline(subject)
    second = pipeline(subject)
    np.testing.assert_array_equal(
        np.asarray(first.label_array), np.asarray(second.label_array)
    )


@pytest.mark.unit
def test_pipeline_without_supervoxelizer_clusters_voxels_directly() -> None:
    """``supervoxelizer=None`` selects the direct-clustering designs."""
    pipeline = _fitted_pipeline(supervoxels=False)
    habitat_map = pipeline(make_subject("new", seed=9))
    labels = set(np.unique(np.asarray(habitat_map.label_array)))
    assert labels <= {0, 1, 2}
    assert labels - {0}


@pytest.mark.unit
def test_voxel_units_wrap_field_as_singleton_partition() -> None:
    """Each voxel becomes its own clustering unit, preserving order."""
    field = make_field("P1", n_voxels=6)
    units = voxel_units(field)
    labels = np.asarray(units.label_array)
    np.testing.assert_array_equal(
        labels[tuple(field.voxel_index.T)], np.arange(1, 7)
    )
    assert units.features.shape == (6, 2)
    np.testing.assert_allclose(units.features.to_numpy(), field.values)


@pytest.mark.unit
def test_pipeline_extract_features_joins_families() -> None:
    """extract_features returns one row joined across all families."""
    pipeline = _fitted_pipeline()
    table = pipeline.extract_features(
        make_subject("new", seed=3),
        [MsiHabitatFeatures(), IthHabitatFeatures(), HabitatVolumeFeatures()],
    )
    assert table.frame.shape[0] == 1
    assert table.id_columns == ("subject",)
    assert "ith_score" in table.feature_columns
    assert "contrast" in table.feature_columns
    assert "habitat_1_volume_fraction" in table.feature_columns
    assert table.frame.iloc[0]["subject"] == "new"


@pytest.mark.unit
def test_pipeline_extract_features_requires_extractors() -> None:
    """An empty extractor list is an explicit error."""
    pipeline = _fitted_pipeline()
    with pytest.raises(HABITAPIError):
        pipeline.extract_features(make_subject("new", seed=3), [])


@pytest.mark.unit
def test_pipeline_spec_covers_every_stage() -> None:
    """The composed fingerprint changes when any stage changes."""
    base = _fitted_pipeline()
    tweaked = SubjectPipeline(
        RawVoxelFeatures(modalities=["T1"]),
        SlicSupervoxelizer(n_supervoxels=16),
        base.habitat_assigner,
    )
    assert base.spec.name == "subject_pipeline"
    assert base.spec.fingerprint() != tweaked.spec.fingerprint()


@pytest.mark.unit
def test_pipeline_requires_core_steps() -> None:
    """Only the supervoxelizer may be omitted."""
    with pytest.raises(HABITAPIError):
        SubjectPipeline(None, None, None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_pipeline_runs_under_cohort_map() -> None:
    """The pipeline is an ordinary subject-level operator for Cohort.map."""
    pipeline = _fitted_pipeline()
    cohort = Cohort([make_subject(f"E{i}", seed=20 + i) for i in range(2)])
    maps = cohort.map(pipeline)
    assert [m.subject_id for m in maps] == ["E0", "E1"]
    assert all(isinstance(m, HabitatMap) for m in maps)


@pytest.mark.unit
def test_pipeline_applies_the_voxel_feature_preprocessor() -> None:
    """A configured voxel chain must actually change the units it feeds.

    The bug this guards against is a chain that is specified, serialised into
    provenance, and never called: the run reports normalisation it did not do.
    """
    subject = make_subject("new", seed=8)
    plain = _fitted_pipeline()
    scaled = SubjectPipeline(
        plain.voxel_feature_extractor,
        plain.supervoxelizer,
        plain.habitat_assigner,
        voxel_feature_preprocessor=SubjectPreprocessingChain(
            build_methods([{"name": "minmax"}])
        ),
    )
    raw_units = plain.units(subject)
    scaled_units = scaled.units(subject)
    assert raw_units.features.shape == scaled_units.features.shape
    # min-max on the voxel features bounds every supervoxel mean to [0, 1],
    # which the unscaled features are not.
    assert scaled_units.features.to_numpy().max() <= 1.0
    assert raw_units.features.to_numpy().max() > 1.0


@pytest.mark.unit
def test_pipeline_applies_the_supervoxel_feature_preprocessor() -> None:
    """A configured supervoxel chain must change the supervoxel features."""
    subject = make_subject("new", seed=12)
    plain = _fitted_pipeline()
    scaled = SubjectPipeline(
        plain.voxel_feature_extractor,
        plain.supervoxelizer,
        plain.habitat_assigner,
        supervoxel_feature_preprocessor=SubjectPreprocessingChain(
            build_methods([{"name": "zscore"}])
        ),
    )
    units = scaled.units(subject)
    # z-scoring per subject centres every feature column on zero.
    np.testing.assert_allclose(
        units.features.to_numpy().mean(axis=0), 0.0, atol=1e-9
    )
    # The partition itself is untouched: describing regions never redraws them.
    np.testing.assert_array_equal(
        np.asarray(units.label_array),
        np.asarray(plain.units(subject).label_array),
    )


@pytest.mark.unit
def test_pipeline_applies_the_fitted_cohort_preprocessor_before_assignment() -> None:
    """The cohort chain is replayed at prediction time, changing the labels.

    Applying a habitat model without the cohort preprocessing it was defined
    with would put the new subject in a different feature space and still
    return plausible labels -- the failure mode that is impossible to notice
    from the output alone.
    """
    subject = make_subject("new", seed=15)
    plain = _fitted_pipeline()
    chain = CohortPreprocessingChain(build_methods([{"name": "minmax"}]))
    chain.fit(plain.units(subject).feature_frame() * 3.0)
    wired = SubjectPipeline(
        plain.voxel_feature_extractor,
        plain.supervoxelizer,
        plain.habitat_assigner,
        cohort_feature_preprocessor=chain,
    )
    assert not np.array_equal(
        np.asarray(plain(subject).label_array),
        np.asarray(wired(subject).label_array),
    )


@pytest.mark.unit
def test_pipeline_spec_records_every_preprocessing_chain() -> None:
    """Each chain enters the fingerprint, so a run is identifiable by it."""
    plain = _fitted_pipeline()
    payload = plain.spec.params
    for slot in (
        "voxel_feature_preprocessor",
        "supervoxel_feature_preprocessor",
        "cohort_feature_preprocessor",
    ):
        assert slot in payload
        assert payload[slot] is None
    wired = SubjectPipeline(
        plain.voxel_feature_extractor,
        plain.supervoxelizer,
        plain.habitat_assigner,
        voxel_feature_preprocessor=SubjectPreprocessingChain(
            build_methods([{"name": "minmax"}])
        ),
    )
    assert wired.spec.fingerprint() != plain.spec.fingerprint()
    steps = wired.spec.params["voxel_feature_preprocessor"]["params"]["steps"]
    assert [step["name"] for step in steps] == ["impute", "minmax"]


@pytest.mark.unit
def test_supervoxel_preprocessor_requires_a_supervoxelizer() -> None:
    """Without supervoxels there is only one matrix, so the slot is refused."""
    plain = _fitted_pipeline(supervoxels=False)
    with pytest.raises(HABITAPIError, match="voxel_feature_preprocessor instead"):
        SubjectPipeline(
            plain.voxel_feature_extractor,
            None,
            plain.habitat_assigner,
            supervoxel_feature_preprocessor=SubjectPreprocessingChain(
                build_methods([{"name": "minmax"}])
            ),
        )
