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

"""Per-subject resume in the habitat study paths (stage-5 wiring).

A ``CheckpointStore`` passed to :class:`~habit.recipes.study.Study` (fit or
predict) is forwarded to every ``Cohort.map`` the study runs. Cache keys
carry their validity scope inside the key string, so correctness never
depends on store-level metadata:

* clustering units key on the SPEC fingerprint -- they depend on one
  subject and the spec alone, so a growing cohort reuses earlier subjects;
* habitat labels key on the fitted model's ``model_id`` -- which already
  embeds the spec fingerprint and the defining cohort's digest, so a
  refitted or different definition never reuses stale labels;
* one-step subjects key on the spec fingerprint only, matching the
  design's subject independence.

Everything below runs on the in-memory synthetic cohort and finishes in
seconds; the "second run" proofs use a sabotaged cohort whose subjects
carry no images at all, so any cache miss would fail loudly instead of
silently recomputing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from habit.contracts.subject import Cohort, Subject
from habit.datasets import make_synthetic_cohort
from habit.execution.checkpoint import CheckpointStore
from habit.recipes.result import StudyResult
from habit.recipes.study import Study
from habit.spec.specs import HabitatSpec, Spec

#: Volume shape for every cohort in this module; small enough that SLIC
#: and kmeans are instantaneous, large enough for three z-band habitats.
_SHAPE = (16, 16, 16)


def _spec(*, two_step: bool = True, n_habitats: int = 3) -> HabitatSpec:
    """
    Build a fast habitat spec with a fixed habitat count.

    Args:
        two_step: When ``True`` include a SLIC supervoxel stage.
        n_habitats: Fixed cluster count for the kmeans fitter.

    Returns:
        A fully seeded :class:`~habit.spec.specs.HabitatSpec`.
    """
    return HabitatSpec(
        name="resume_test",
        voxel_feature_extractor=Spec(
            name="raw",
            params={"modalities": ["T1", "T2"], "roi": "tumor"},
        ),
        supervoxelizer=(
            Spec(name="slic", params={"n_supervoxels": 8, "compactness": 5.0})
            if two_step
            else None
        ),
        habitat_model_fitter=Spec(
            name="kmeans",
            params={"n_habitats": n_habitats, "n_init": 2},
        ),
        habitat_assigner=Spec(name="nearest_centroid", params={}),
        habitat_features=(Spec(name="volume"),),
        random_seed=0,
    )


def _cohort(n_subjects: int = 4) -> Cohort:
    """Return a small deterministic synthetic cohort."""
    return make_synthetic_cohort(
        n_subjects=n_subjects,
        modalities=("T1", "T2"),
        shape=_SHAPE,
        n_subregions=3,
        rng=0,
    )


def _sabotaged_cohort(cohort: Cohort) -> Cohort:
    """
    Return a cohort with identical subject ids but no usable inputs.

    Subject ids are all a resumed run needs: cached computations never
    touch images. Any cache miss, however, would fail inside the pipeline
    (there is no ROI to read), so a completed sabotaged run proves that
    every subject came back from the checkpoint.
    """
    return Cohort(
        [
            Subject(subject_id=subject.subject_id, images={}, masks={})
            for subject in cohort
        ],
        name=cohort.name,
    )


def _assert_same_study(first: StudyResult, second: StudyResult) -> None:
    """Assert voxel-wise and tabular equality between two study results."""
    first_ids = [m.subject_id for m in first.habitat_maps]
    second_ids = [m.subject_id for m in second.habitat_maps]
    assert first_ids == second_ids
    for first_map, second_map in zip(first.habitat_maps, second.habitat_maps):
        assert first_map.model_id == second_map.model_id
        assert np.array_equal(first_map.label_array, second_map.label_array)
    pd.testing.assert_frame_equal(first.features.frame, second.features.frame)


@pytest.mark.unit
def test_two_step_resume_reuses_every_subject(tmp_path: Path) -> None:
    """A resumed two-step run recomputes nothing and reproduces the maps."""
    store = CheckpointStore(tmp_path / "ckpt")
    cohort = _cohort(4)

    first = Study(spec=_spec(), design="two_step").fit_predict(cohort, checkpoint=store)
    # Four units entries plus four label entries.
    assert len(store) == 8

    second = Study(spec=_spec(), design="two_step").fit_predict(
        _sabotaged_cohort(cohort), checkpoint=store
    )

    assert len(store) == 8
    assert first.habitat_model is not None and second.habitat_model is not None
    assert first.habitat_model.model_id == second.habitat_model.model_id
    _assert_same_study(first, second)


@pytest.mark.unit
def test_direct_pooling_resume_reuses_every_subject(tmp_path: Path) -> None:
    """Direct pooling resumes identically (voxel units, no supervoxels)."""
    store = CheckpointStore(tmp_path / "ckpt")
    cohort = _cohort(2)
    spec = _spec(two_step=False)

    first = Study(spec=spec, design="direct_pooling").fit_predict(cohort, checkpoint=store)
    assert len(store) == 4

    second = Study(spec=spec, design="direct_pooling").fit_predict(
        _sabotaged_cohort(cohort), checkpoint=store
    )

    assert len(store) == 4
    _assert_same_study(first, second)


@pytest.mark.unit
def test_one_step_resume_reuses_every_subject(tmp_path: Path) -> None:
    """One-step caches one whole per-subject computation per entry."""
    store = CheckpointStore(tmp_path / "ckpt")
    cohort = _cohort(3)
    spec = _spec(two_step=False)

    first = Study(spec=spec, design="one_step").fit_predict(cohort, checkpoint=store)
    assert len(store) == 3

    second = Study(spec=spec, design="one_step").fit_predict(
        _sabotaged_cohort(cohort), checkpoint=store
    )

    assert len(store) == 3
    assert set(second.subject_models) == set(first.subject_models)
    _assert_same_study(first, second)


@pytest.mark.unit
def test_spec_change_invalidates_cached_entries(tmp_path: Path) -> None:
    """A changed spec never reads stale entries; the old ones stay reachable."""
    store = CheckpointStore(tmp_path / "ckpt")
    cohort = _cohort(2)
    spec_a = _spec(n_habitats=3)
    spec_b = _spec(n_habitats=2)

    first_a = Study(spec=spec_a, design="two_step").fit_predict(cohort, checkpoint=store)
    first_b = Study(spec=spec_b, design="two_step").fit_predict(cohort, checkpoint=store)

    # Two subjects x two stages x two specs: B recomputed everything.
    assert len(store) == 8
    assert first_b.habitat_model is not None and first_a.habitat_model is not None
    assert first_b.habitat_model.n_habitats == 2
    assert first_b.habitat_model.model_id != first_a.habitat_model.model_id

    # Switching back to spec A hits A's entries again.
    second_a = Study(spec=spec_a, design="two_step").fit_predict(
        _sabotaged_cohort(cohort), checkpoint=store
    )
    assert len(store) == 8
    _assert_same_study(first_a, second_a)


@pytest.mark.unit
def test_cohort_growth_reuses_units_but_relabels(tmp_path: Path) -> None:
    """Adding a subject reuses its predecessors' units, not their labels."""
    store = CheckpointStore(tmp_path / "ckpt")
    cohort3 = _cohort(3)
    cohort4 = _cohort(4)
    spec = _spec()

    Study(spec=spec, design="two_step").fit_predict(cohort3, checkpoint=store)
    assert len(store) == 6

    # The 4-subject model differs, so labels recompute (4 new entries),
    # while the first three subjects' units are reused (1 new units entry).
    grown = Study(spec=spec, design="two_step").fit_predict(cohort4, checkpoint=store)
    assert len(store) == 6 + 1 + 4
    assert grown.habitat_model is not None


@pytest.mark.unit
def test_apply_habitat_model_scopes_entries_by_model(tmp_path: Path) -> None:
    """Two definitions applied through one store never share entries."""
    store = CheckpointStore(tmp_path / "ckpt")
    cohort4 = _cohort(4)
    cohort3 = _cohort(3)
    spec = _spec()

    model_a = Study(spec=spec, design="two_step").fit_predict(cohort4).habitat_model
    model_b = Study(spec=spec, design="two_step").fit_predict(cohort3).habitat_model
    assert model_a is not None and model_b is not None
    assert model_a.model_id != model_b.model_id

    first_a = Study.from_model(model_a, spec).predict(cohort4, checkpoint=store)
    assert len(store) == 4

    first_b = Study.from_model(model_b, spec).predict(cohort4, checkpoint=store)
    # A different model_id must not hit model A's labels.
    assert len(store) == 8
    assert all(m.model_id == model_b.model_id for m in first_b.habitat_maps)

    second_a = Study.from_model(model_a, spec).predict(
        _sabotaged_cohort(cohort4), checkpoint=store
    )
    assert len(store) == 8
    _assert_same_study(first_a, second_a)
