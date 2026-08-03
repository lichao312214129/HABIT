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
"""Contract tests for FeatureTable, RunManifest and StudyResult."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from habit.api.exceptions import HABITAPIError
from habit.contracts import (
    CohortFingerprint,
    FeatureTable,
    Geometry,
    HabitatMap,
    HabitatModel,
    Provenance,
    RunManifest,
    StudyResult,
)


def _table(frame: pd.DataFrame | None = None, **overrides) -> FeatureTable:
    """Build a small valid feature table."""
    frame = frame if frame is not None else pd.DataFrame(
        {"subject": ["a", "b"], "f1": [1.0, 2.0], "f2": [3.0, 4.0], "y": [0, 1]}
    )
    return FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=("f1", "f2"),
        outcome_column="y",
        provenance=Provenance.source("test"),
        **overrides,
    )


@pytest.mark.unit
def test_feature_table_validates_declared_columns() -> None:
    """Declaring a missing column fails fast instead of leaking later."""
    with pytest.raises(HABITAPIError):
        FeatureTable(
            frame=pd.DataFrame({"subject": ["a"]}),
            id_columns=("subject",),
            feature_columns=("missing",),
        )


@pytest.mark.unit
def test_feature_matrix_excludes_ids_and_outcome() -> None:
    """The model matrix carries only feature columns, indexed by ids."""
    matrix = _table().feature_matrix()
    assert list(matrix.columns) == ["f1", "f2"]
    assert list(matrix.index.names) == ["subject"]
    assert matrix.loc["b", "f2"] == 4.0


@pytest.mark.unit
def test_feature_table_join_combines_families() -> None:
    """join merges on shared id columns and unions the feature columns."""
    left = _table()
    right = FeatureTable(
        frame=pd.DataFrame({"subject": ["a", "b"], "msi": [0.1, 0.2]}),
        id_columns=("subject",),
        feature_columns=("msi",),
        provenance=Provenance.source("msi"),
    )
    joined = left.join(right)
    assert joined.feature_columns == ("f1", "f2", "msi")
    assert joined.frame.loc[joined.frame["subject"] == "a", "msi"].iloc[0] == 0.1
    assert joined.provenance is not None
    assert len(joined.provenance.inputs) == 2


@pytest.mark.unit
def test_feature_table_join_rejects_mismatched_ids_and_overlap() -> None:
    """Id mismatch or duplicate feature columns are explicit errors."""
    left = _table()
    mismatched = FeatureTable(
        frame=pd.DataFrame({"case": ["a"], "g": [1.0]}),
        id_columns=("case",),
        feature_columns=("g",),
    )
    with pytest.raises(HABITAPIError):
        left.join(mismatched)
    overlapping = FeatureTable(
        frame=pd.DataFrame({"subject": ["a"], "f1": [9.0]}),
        id_columns=("subject",),
        feature_columns=("f1",),
    )
    with pytest.raises(HABITAPIError):
        left.join(overlapping)


def _manifest() -> RunManifest:
    """Build a manifest whose provenance DAG has a seed and a failure."""
    root = Provenance.source("raw")
    fitted = root.derive(
        produced_by="habitat_model_fitter.kmeans",
        spec_fingerprint="abc",
        random_seed=42,
    )
    return RunManifest(
        spec_payload={"habitat_model_fitter": {"name": "kmeans"}},
        provenance=fitted,
        subject_outcomes={"a": "success", "b": "RuntimeError: boom"},
        started_at="2026-08-03T00:00:00+00:00",
        finished_at="2026-08-03T00:01:00+00:00",
    )


@pytest.mark.unit
def test_manifest_reports_versions_seeds_and_exclusions() -> None:
    """The manifest derives its facts from the provenance DAG only."""
    manifest = _manifest()

    assert manifest.software_versions()["habit"]
    assert manifest.random_seeds() == {"habitat_model_fitter.kmeans": 42}

    methods = manifest.describe_methods()
    assert "HABIT" in methods
    assert "habitat_model_fitter.kmeans" in methods
    assert "42" in methods
    assert "b" in methods and "excluded" in methods

    with pytest.raises(HABITAPIError):
        manifest.describe_methods(style="imaginary")


@pytest.mark.unit
def test_manifest_checklist_never_fakes_unverifiable_items() -> None:
    """Items HABIT cannot evidence are marked needs_human_answer."""
    checklist = _manifest().checklist("CLEAR")
    statuses = dict(zip(checklist["item"], checklist["status"]))

    assert statuses["software_version"] == "evidenced"
    assert statuses["random_seeds"] == "evidenced"
    assert statuses["clinical_cohort_description"] == "needs_human_answer"
    with pytest.raises(HABITAPIError):
        _manifest().checklist("NOT_A_STANDARD")


@pytest.mark.unit
def test_manifest_to_json_roundtrip(tmp_path: Path) -> None:
    """to_json returns text and optionally writes it."""
    manifest = _manifest()
    text = manifest.to_json(tmp_path / "nested" / "manifest.json")

    payload = json.loads(text)
    assert payload["subject_outcomes"]["b"].startswith("RuntimeError")
    assert (tmp_path / "nested" / "manifest.json").is_file()


@pytest.mark.unit
def test_study_result_save_writes_artefacts(tmp_path: Path) -> None:
    """StudyResult.save is the single explicit act of writing to disk."""
    provenance = Provenance.source("study")
    model = HabitatModel(
        model_id="m",
        n_habitats=1,
        feature_names=("f1",),
        centroids=np.zeros((1, 1)),
        preprocessing_state={},
        spec_payload={},
        cohort_fingerprint=CohortFingerprint(
            n_subjects=1, modalities=("T1",), subject_id_digest="d"
        ),
        provenance=provenance,
    )
    result = StudyResult(
        habitat_model=model,
        pipeline=object(),
        features=_table(),
        habitat_maps=(
            HabitatMap(
                subject_id="a",
                label_array=np.ones((2, 2, 2), dtype=np.int32),
                geometry=Geometry.from_array((2, 2, 2)),
                model_id="m",
                habitat_ids=(1,),
                provenance=provenance,
            ),
        ),
        manifest=_manifest(),
    )
    out = result.save(tmp_path / "study")

    assert (out / "habitat_model.habitatmodel").is_file()
    assert (out / "habitat_features.csv").is_file()
    assert (out / "run_manifest.json").is_file()
    reloaded = HabitatModel.load(out / "habitat_model.habitatmodel")
    assert reloaded.model_id == "m"
