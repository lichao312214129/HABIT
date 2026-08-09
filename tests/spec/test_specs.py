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
"""Contract tests for the Spec / HabitatSpec / RunPolicy tripartition."""

from __future__ import annotations

import numpy as np
import pytest

from habit.api.exceptions import HABITAPIError
from habit.spec import (
    HabitatSpec,
    MLSpec,
    RunPolicy,
    Spec,
    load_habitat_spec,
    load_run_policy,
    save_habitat_spec,
    save_run_policy,
)


def _habitat_spec() -> HabitatSpec:
    """Build a representative composed specification."""
    return HabitatSpec(
        name="demo",
        voxel_feature_extractor=Spec(name="raw", params={"modalities": ["T1"]}),
        supervoxelizer=Spec(name="slic", params={"n_supervoxels": 50}),
        habitat_model_fitter=Spec(name="kmeans", params={"n_habitats": 3}),
        habitat_assigner=Spec(name="nearest_centroid"),
        habitat_features=(Spec(name="msi"), Spec(name="ith_score")),
    )


@pytest.mark.unit
def test_spec_fingerprint_is_stable_and_param_sensitive() -> None:
    """Equal specs hash equally; any param deviation changes the hash."""
    first = Spec(name="slic", params={"n_supervoxels": 100, "compactness": 10.0})
    second = Spec(name="slic", params={"compactness": 10.0, "n_supervoxels": 100})
    different = Spec(name="slic", params={"n_supervoxels": 101})
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != different.fingerprint()


@pytest.mark.unit
def test_spec_dict_roundtrip_and_numpy_normalisation() -> None:
    """NumPy scalars serialise to plain JSON values and back."""
    spec = Spec(name="kmeans", params={"n_habitats": np.int64(4), "tol": np.float64(1e-4)})
    payload = spec.to_dict()
    assert payload["params"] == {"n_habitats": 4, "tol": 0.0001}
    assert Spec.from_dict(payload) == spec


@pytest.mark.unit
def test_spec_validation_errors() -> None:
    """Blank names and nameless payloads are rejected at the boundary."""
    with pytest.raises(HABITAPIError):
        Spec(name="  ")
    with pytest.raises(HABITAPIError):
        Spec.from_dict({"params": {}})


@pytest.mark.unit
def test_habitat_spec_roundtrip_and_component_specs() -> None:
    """The composed spec survives a dict roundtrip with domain-verbatim keys."""
    spec = _habitat_spec()
    payload = spec.to_dict()
    assert set(payload) == {
        "name",
        "version",
        "voxel_feature_extractor",
        "supervoxelizer",
        "supervoxel_feature_extractor",
        "habitat_model_fitter",
        "habitat_assigner",
        "habitat_features",
        "voxel_feature_preprocessors",
        "supervoxel_feature_preprocessors",
        "cohort_feature_preprocessors",
        "random_seed",
    }
    restored = HabitatSpec.from_dict(payload)
    assert restored == spec
    assert restored.on_geometry_mismatch == "resample_mask"
    assert "on_geometry_mismatch" not in payload
    assert restored.component_specs()["supervoxelizer"].name == "slic"
    assert [s.name for s in restored.habitat_features] == ["msi", "ith_score"]


@pytest.mark.unit
def test_habitat_spec_strict_geometry_policy_roundtrips_and_fingerprints() -> None:
    """Opting into strict geometry changes the fingerprint; default stays omitted."""
    base = _habitat_spec()
    strict = HabitatSpec.from_dict(
        {**base.to_dict(), "on_geometry_mismatch": "strict"}
    )
    assert strict.on_geometry_mismatch == "strict"
    assert strict.to_dict()["on_geometry_mismatch"] == "strict"
    assert strict.fingerprint() != base.fingerprint()
    assert HabitatSpec.from_dict(strict.to_dict()) == strict


@pytest.mark.unit
def test_habitat_spec_postprocess_slots_roundtrip_and_fingerprint() -> None:
    """Enabled postprocess Specs round-trip; defaults stay omitted from payload."""
    base = _habitat_spec()
    assert "postprocess_habitat" not in base.to_dict()
    assert "postprocess_supervoxel" not in base.to_dict()
    with_pp = HabitatSpec(
        name=base.name,
        voxel_feature_extractor=base.voxel_feature_extractor,
        supervoxelizer=base.supervoxelizer,
        habitat_model_fitter=base.habitat_model_fitter,
        habitat_assigner=base.habitat_assigner,
        habitat_features=base.habitat_features,
        postprocess_habitat=Spec(
            name="connected_components",
            params={"min_component_size": 30, "connectivity": 1},
        ),
    )
    payload = with_pp.to_dict()
    assert payload["postprocess_habitat"]["name"] == "connected_components"
    assert "postprocess_supervoxel" not in payload
    restored = HabitatSpec.from_dict(payload)
    assert restored == with_pp
    assert restored.fingerprint() != base.fingerprint()


@pytest.mark.unit
def test_habitat_spec_preprocessor_chains_and_random_seed() -> None:
    """The three preprocessing chains and the seed roundtrip and fingerprint."""
    spec = HabitatSpec(
        name="chains",
        voxel_feature_extractor=Spec(name="raw", params={"modalities": ["T1"]}),
        supervoxelizer=Spec(name="slic"),
        habitat_model_fitter=Spec(name="kmeans"),
        habitat_assigner=Spec(name="nearest_centroid"),
        voxel_feature_preprocessors=(
            Spec(name="winsorize", params={"winsor_limits": [0.05, 0.05]}),
            Spec(name="minmax"),
        ),
        supervoxel_feature_preprocessors=(Spec(name="zscore"),),
        cohort_feature_preprocessors=(
            Spec(name="binning", params={"n_bins": 10, "bin_strategy": "uniform"}),
        ),
        random_seed=42,
    )
    restored = HabitatSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert [s.name for s in restored.voxel_feature_preprocessors] == [
        "winsorize",
        "minmax",
    ]
    assert [s.name for s in restored.supervoxel_feature_preprocessors] == ["zscore"]
    assert [s.name for s in restored.cohort_feature_preprocessors] == ["binning"]
    assert restored.random_seed == 42
    # The three chains are independent slots: moving a step between them is a
    # different analysis, so it must change the fingerprint.
    moved = HabitatSpec.from_dict(
        {
            **spec.to_dict(),
            "voxel_feature_preprocessors": [],
            "cohort_feature_preprocessors": [
                *spec.to_dict()["cohort_feature_preprocessors"],
                *spec.to_dict()["voxel_feature_preprocessors"],
            ],
        }
    )
    assert moved.fingerprint() != spec.fingerprint()
    # The seed is scientific: changing it must change the fingerprint.
    reseeded = HabitatSpec.from_dict({**spec.to_dict(), "random_seed": 7})
    assert reseeded.fingerprint() != spec.fingerprint()


@pytest.mark.unit
def test_habitat_spec_allows_none_supervoxelizer_for_direct_designs() -> None:
    """A missing supervoxel spec selects the direct-clustering designs."""
    spec = HabitatSpec(
        name="direct",
        voxel_feature_extractor=Spec(name="raw", params={"modalities": ["T1"]}),
        supervoxelizer=None,
        habitat_model_fitter=Spec(name="kmeans"),
        habitat_assigner=Spec(name="nearest_centroid"),
    )
    payload = spec.to_dict()
    assert payload["supervoxelizer"] is None
    restored = HabitatSpec.from_dict(payload)
    assert restored.supervoxelizer is None


@pytest.mark.unit
def test_habitat_spec_requires_core_components() -> None:
    """The voxel/fitter/assigner slots are mandatory."""
    with pytest.raises(HABITAPIError):
        HabitatSpec(
            name="broken",
            voxel_feature_extractor=None,  # type: ignore[arg-type]
            supervoxelizer=None,
            habitat_model_fitter=Spec(name="kmeans"),
            habitat_assigner=Spec(name="nearest_centroid"),
        )


@pytest.mark.unit
def test_habitat_spec_fingerprint_changes_with_any_stage() -> None:
    """The composed fingerprint binds every stage's parameters."""
    base = _habitat_spec()
    tweaked = HabitatSpec(
        name="demo",
        voxel_feature_extractor=Spec(name="raw", params={"modalities": ["T1", "T2"]}),
        supervoxelizer=Spec(name="slic", params={"n_supervoxels": 50}),
        habitat_model_fitter=Spec(name="kmeans", params={"n_habitats": 3}),
        habitat_assigner=Spec(name="nearest_centroid"),
        habitat_features=(Spec(name="msi"), Spec(name="ith_score")),
    )
    assert base.fingerprint() != tweaked.fingerprint()


@pytest.mark.unit
def test_habitat_spec_describe_methods_states_every_configured_step() -> None:
    """The planned-methods paragraph names every component and its params."""
    spec = _habitat_spec()
    text = spec.describe_methods()

    assert text.startswith("A habitat imaging analysis was designed with HABIT")
    assert "'demo'" in text
    # Every configured step appears, with its parameters, in pipeline order.
    assert text.index("voxel feature extraction with raw") < text.index(
        "supervoxelization with slic"
    ) < text.index("habitat model fitting with kmeans") < text.index(
        "habitat assignment with nearest_centroid"
    ) < text.index("habitat feature families: msi")
    assert "n_supervoxels=50" in text
    assert "n_habitats=3" in text
    assert "modalities=['T1']" in text
    assert "aligned onto the image grid" in text
    # No seed was configured, so none is promised.
    assert "random seed" not in text.lower()


@pytest.mark.unit
def test_habitat_spec_describe_methods_styles_differ_only_in_ordering() -> None:
    """Radiology opens with the design sentence; nature closes with it."""
    spec = HabitatSpec(
        name="styled",
        voxel_feature_extractor=Spec(name="raw", params={"modalities": ["T1"]}),
        supervoxelizer=None,
        habitat_model_fitter=Spec(name="kmeans"),
        habitat_assigner=Spec(name="nearest_centroid"),
        random_seed=42,
    )
    radiology = spec.describe_methods(style="radiology")
    nature = spec.describe_methods(style="nature")

    assert radiology.startswith("A habitat imaging analysis was designed with HABIT")
    assert nature.endswith("The analysis was designed with HABIT.")
    # The direct-clustering design is stated honestly, and the seed is fixed.
    assert "direct voxel clustering (no supervoxelization)" in radiology
    assert "Random seed 42 is fixed" in nature

    with pytest.raises(HABITAPIError):
        spec.describe_methods(style="imaginary")


def _dataflow_spec(**overrides: object) -> HabitatSpec:
    """Build a minimal valid spec; overrides select the declared dataflow."""
    fields = {
        "name": "dataflow",
        "voxel_feature_extractor": Spec(name="raw", params={"modalities": ["T1"]}),
        "supervoxelizer": None,
        "habitat_model_fitter": Spec(name="kmeans"),
        "habitat_assigner": Spec(name="nearest_centroid"),
    }
    fields.update(overrides)
    return HabitatSpec(**fields)  # type: ignore[arg-type]


@pytest.mark.unit
def test_habitat_spec_pooling_value_domain() -> None:
    """Only 'cohort' / 'none' / None are accepted, case-normalised."""
    assert _dataflow_spec().pooling is None
    assert _dataflow_spec(pooling="cohort").pooling == "cohort"
    assert _dataflow_spec(pooling="NONE").pooling == "none"
    with pytest.raises(HABITAPIError, match="pooling"):
        _dataflow_spec(pooling="sometimes")


@pytest.mark.unit
def test_habitat_spec_definition_level_is_derived() -> None:
    """The definition level follows the declared dataflow, never free text."""
    assert _dataflow_spec().definition_level == "cohort"
    assert _dataflow_spec(pooling="cohort").definition_level == "cohort"
    assert _dataflow_spec(pooling="none").definition_level == "subject"


@pytest.mark.unit
def test_habitat_spec_pooling_fingerprint_stability() -> None:
    """Undeclared and explicit-cohort share one fingerprint; 'none' is recorded."""
    undeclared = _dataflow_spec()
    explicit_cohort = _dataflow_spec(pooling="cohort")
    subject_level = _dataflow_spec(pooling="none")

    assert undeclared.fingerprint() == explicit_cohort.fingerprint()
    assert "pooling" not in undeclared.to_dict()
    assert "pooling" not in explicit_cohort.to_dict()

    payload = subject_level.to_dict()
    assert payload["pooling"] == "none"
    assert payload["definition_level"] == "subject"
    assert subject_level.fingerprint() != undeclared.fingerprint()
    # The recorded dataflow round-trips losslessly.
    restored = HabitatSpec.from_dict(payload)
    assert restored.pooling == "none"
    assert restored.fingerprint() == subject_level.fingerprint()


@pytest.mark.unit
def test_habitat_spec_effective_dict_states_the_dataflow() -> None:
    """YAML export always spells out pooling and the derived level."""
    effective = _dataflow_spec().to_effective_dict()
    assert effective["pooling"] == "cohort"
    assert effective["definition_level"] == "cohort"
    assert "pooling" not in _dataflow_spec().to_dict()

    subject_level = _dataflow_spec(pooling="none").to_effective_dict()
    assert subject_level["pooling"] == "none"
    assert subject_level["definition_level"] == "subject"

    # Round-trip through the effective form keeps the fingerprint stable.
    for spec in (_dataflow_spec(), _dataflow_spec(pooling="cohort"),
                 _dataflow_spec(pooling="none")):
        restored = HabitatSpec.from_dict(spec.to_effective_dict())
        assert restored.fingerprint() == spec.fingerprint()


@pytest.mark.unit
def test_habitat_spec_from_dict_rejects_a_contradictory_level() -> None:
    """definition_level is derived; a disagreeing document is rejected."""
    payload = _dataflow_spec(pooling="none").to_dict()
    payload["definition_level"] = "cohort"
    with pytest.raises(HABITAPIError, match="definition_level"):
        HabitatSpec.from_dict(payload)


@pytest.mark.unit
def test_habitat_spec_validate_dataflow() -> None:
    """Subject-level definition forbids supervoxels and cohort chains."""
    _dataflow_spec().validate_dataflow()
    _dataflow_spec(pooling="cohort").validate_dataflow()
    _dataflow_spec(pooling="none").validate_dataflow()

    with_supervoxels = _dataflow_spec(
        pooling="none", supervoxelizer=Spec(name="kmeans")
    )
    with pytest.raises(HABITAPIError, match="supervoxelizer"):
        with_supervoxels.validate_dataflow()

    with_cohort_chain = _dataflow_spec(
        pooling="none",
        cohort_feature_preprocessors=(Spec(name="zscore"),),
    )
    with pytest.raises(HABITAPIError, match="cohort_feature_preprocessors"):
        with_cohort_chain.validate_dataflow()


@pytest.mark.unit
def test_habitat_spec_describe_methods_states_subject_level_dataflow() -> None:
    """Only the subject-level design adds a dataflow sentence to the prose."""
    cohort_text = _dataflow_spec().describe_methods()
    assert "cross-subject pooling" not in cohort_text

    subject_text = _dataflow_spec(pooling="none").describe_methods()
    assert "no cross-subject pooling" in subject_text
    assert "not comparable across subjects" in subject_text


@pytest.mark.unit
def test_run_policy_defaults_and_validation() -> None:
    """Policy defaults are serial execution with continue-on-failure."""
    policy = RunPolicy()
    assert policy.workers == 1
    assert policy.on_subject_failure == "continue"
    assert policy.backend == "serial"
    assert policy.resume is True
    with pytest.raises(HABITAPIError):
        RunPolicy(workers=0)
    with pytest.raises(HABITAPIError):
        RunPolicy(on_subject_failure="ignore")
    with pytest.raises(HABITAPIError):
        RunPolicy(backend="dask")
    with pytest.raises(HABITAPIError):
        RunPolicy(subject_timeout_sec=0)
    with pytest.raises(HABITAPIError):
        RunPolicy(parallel_mode="thread")


@pytest.mark.unit
def test_run_policy_dict_roundtrip() -> None:
    """Policies tolerate partial payloads and roundtrip fully."""
    policy = RunPolicy.from_dict({"workers": 4, "backend": "process"})
    assert policy.workers == 4
    assert policy.on_subject_failure == "continue"
    assert RunPolicy.from_dict(policy.to_dict()) == policy


@pytest.mark.unit
def test_run_policy_full_execution_surface_roundtrip() -> None:
    """The documented execution fields survive a dict roundtrip."""
    policy = RunPolicy(
        workers=8,
        backend="process",
        subject_timeout_sec=1800.0,
        subject_spawn_timeout_sec=None,
        graceful_shutdown_sec=20.0,
        on_subject_failure="fail_fast",
        oom_backoff=True,
        oom_reduce_workers_by=2,
        cap_workers_to_gpu_pool=True,
        resume=False,
        checkpoint_dir="ckpt",
        parallel_mode="isolated",
        auto_retry_rounds=0,
        retry_failed_subjects=True,
        force_rerun_subjects=("S01", "S02"),
        clear_checkpoint_on_success=True,
        strict_checkpoint_hash=True,
    )
    restored = RunPolicy.from_dict(policy.to_dict())
    assert restored == policy
    assert restored.force_rerun_subjects == ("S01", "S02")
    with pytest.raises(HABITAPIError):
        RunPolicy.from_dict({"workers": 2, "processes": 4})  # legacy key rejected


@pytest.mark.unit
def test_yaml_roundtrip_for_spec_and_policy(tmp_path) -> None:
    """YAML is a faithful isomorphism for both specification kinds."""
    spec = _habitat_spec()
    spec_path = save_habitat_spec(spec, tmp_path / "spec.yaml")
    assert load_habitat_spec(spec_path) == spec

    policy = RunPolicy(workers=2, checkpoint_dir=str(tmp_path / "ckpt"))
    policy_path = save_run_policy(policy, tmp_path / "policy.yaml")
    assert load_run_policy(policy_path) == policy


@pytest.mark.unit
def test_yaml_loader_rejects_non_mapping(tmp_path) -> None:
    """A YAML list at the top level is a clear error, not a crash later."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(HABITAPIError):
        load_habitat_spec(bad)
    with pytest.raises(FileNotFoundError):
        load_run_policy(tmp_path / "missing.yaml")


# ---------------------------------------------------------------------------
# MLSpec
# ---------------------------------------------------------------------------


def _ml_spec() -> MLSpec:
    """Build a representative tabular modelling specification."""
    return MLSpec(
        name="demo_ml",
        classifier=Spec(name="LogisticRegression", params={"C": 1.0}),
        table_preprocessors=(Spec(name="zscore"),),
        feature_selectors=(Spec(name="variance", params={"threshold": 0.01}),),
        metrics=(Spec(name="accuracy"), Spec(name="auc")),
        random_seed=42,
    )


@pytest.mark.unit
def test_ml_spec_roundtrip() -> None:
    """The ML spec survives a dict roundtrip with domain-verbatim keys."""
    spec = _ml_spec()
    payload = spec.to_dict()
    assert set(payload) == {
        "name",
        "version",
        "pre_preprocessing_feature_selectors",
        "table_preprocessors",
        "feature_selectors",
        "classifier",
        "metrics",
        "random_seed",
    }
    restored = MLSpec.from_dict(payload)
    assert restored == spec
    assert restored.classifier.name == "LogisticRegression"
    assert [s.name for s in restored.table_preprocessors] == ["zscore"]
    assert [s.name for s in restored.feature_selectors] == ["variance"]
    assert [s.name for s in restored.metrics] == ["accuracy", "auc"]
    assert restored.random_seed == 42


@pytest.mark.unit
def test_ml_spec_fingerprints_selector_stage_assignment() -> None:
    """Two specs differing only in selection stage hash differently."""
    selector = Spec(name="variance", params={"threshold": 0.01})
    pre = MLSpec.from_dict(
        {
            **_ml_spec().to_dict(),
            "pre_preprocessing_feature_selectors": [selector.to_dict()],
            "feature_selectors": [],
        }
    )
    post = MLSpec.from_dict(
        {
            **_ml_spec().to_dict(),
            "pre_preprocessing_feature_selectors": [],
            "feature_selectors": [selector.to_dict()],
        }
    )
    assert pre.pre_preprocessing_feature_selectors == (selector,)
    assert pre.feature_selectors == ()
    assert post.pre_preprocessing_feature_selectors == ()
    assert post.feature_selectors == (selector,)
    assert pre.fingerprint() != post.fingerprint()
    # The stage chains round-trip through the dict form untouched.
    assert MLSpec.from_dict(pre.to_dict()) == pre
    # And the methods paragraph states the stage in execution order.
    text = pre.describe_methods()
    assert text.index("pre-preprocessing feature selection") < text.index(
        "table preprocessing"
    )


@pytest.mark.unit
def test_ml_spec_fingerprint_is_stable_and_param_sensitive() -> None:
    """Equal ML specs hash equally; any stage deviation changes the hash."""
    assert _ml_spec().fingerprint() == _ml_spec().fingerprint()
    tweaked = MLSpec.from_dict(
        {**_ml_spec().to_dict(), "classifier": {"name": "SVM", "params": {}}}
    )
    assert tweaked.fingerprint() != _ml_spec().fingerprint()
    # The seed is scientific: changing it must change the fingerprint.
    reseeded = MLSpec.from_dict({**_ml_spec().to_dict(), "random_seed": 7})
    assert reseeded.fingerprint() != _ml_spec().fingerprint()


@pytest.mark.unit
def test_ml_spec_requires_classifier_and_typed_chains() -> None:
    """The classifier slot is mandatory and chains hold Specs only."""
    with pytest.raises(HABITAPIError):
        MLSpec(name="broken", classifier=None)  # type: ignore[arg-type]
    with pytest.raises(HABITAPIError):
        MLSpec(name="broken", classifier="SVM")  # type: ignore[arg-type]
    with pytest.raises(HABITAPIError):
        MLSpec(
            name="broken",
            classifier=Spec(name="SVM"),
            metrics=("auc",),  # type: ignore[arg-type]
        )
    with pytest.raises(HABITAPIError):
        MLSpec(name="  ", classifier=Spec(name="SVM"))


@pytest.mark.unit
def test_ml_spec_describe_methods_states_every_configured_step() -> None:
    """The planned-methods paragraph names every chain and the classifier."""
    spec = _ml_spec()
    text = spec.describe_methods()

    assert text.startswith("A machine-learning analysis was designed with HABIT")
    assert "'demo_ml'" in text
    # Chains render in pipeline order: preprocessing, selection, classifier.
    assert text.index("table preprocessing with zscore") < text.index(
        "feature selection with variance"
    ) < text.index("a LogisticRegression classifier")
    assert "threshold=0.01" in text
    assert "evaluation metrics: accuracy, auc" in text
    assert "Random seed 42 is fixed" in text

    nature = spec.describe_methods(style="nature")
    assert nature.endswith("The analysis was designed with HABIT.")
    with pytest.raises(HABITAPIError):
        spec.describe_methods(style="imaginary")


@pytest.mark.unit
def test_ml_spec_defaults_to_empty_chains() -> None:
    """Only the classifier is required; chains and seed default to unset."""
    spec = MLSpec(name="bare", classifier=Spec(name="SVM"))
    assert spec.pre_preprocessing_feature_selectors == ()
    assert spec.table_preprocessors == ()
    assert spec.feature_selectors == ()
    assert spec.metrics == ()
    assert spec.random_seed is None
    assert MLSpec.from_dict(spec.to_dict()) == spec
