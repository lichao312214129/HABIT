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
        "habitat_model_fitter",
        "habitat_assigner",
        "habitat_features",
        "subject_table_preprocessors",
        "group_table_preprocessors",
        "random_seed",
    }
    restored = HabitatSpec.from_dict(payload)
    assert restored == spec
    assert restored.component_specs()["supervoxelizer"].name == "slic"
    assert [s.name for s in restored.habitat_features] == ["msi", "ith_score"]


@pytest.mark.unit
def test_habitat_spec_preprocessor_chains_and_random_seed() -> None:
    """Table-preprocessor chains and the seed roundtrip and fingerprint."""
    spec = HabitatSpec(
        name="chains",
        voxel_feature_extractor=Spec(name="raw", params={"modalities": ["T1"]}),
        supervoxelizer=Spec(name="slic"),
        habitat_model_fitter=Spec(name="kmeans"),
        habitat_assigner=Spec(name="nearest_centroid"),
        subject_table_preprocessors=(
            Spec(name="winsorize", params={"winsor_limits": [0.05, 0.05]}),
            Spec(name="minmax"),
        ),
        group_table_preprocessors=(
            Spec(name="binning", params={"n_bins": 10, "bin_strategy": "uniform"}),
        ),
        random_seed=42,
    )
    restored = HabitatSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert [s.name for s in restored.subject_table_preprocessors] == ["winsorize", "minmax"]
    assert [s.name for s in restored.group_table_preprocessors] == ["binning"]
    assert restored.random_seed == 42
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
    # No seed was configured, so none is promised.
    assert "seed" not in text.lower()


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
