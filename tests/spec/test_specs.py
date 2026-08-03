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
        habitat_features=(Spec(name="msi"), Spec(name="ith")),
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
    }
    restored = HabitatSpec.from_dict(payload)
    assert restored == spec
    assert restored.component_specs()["supervoxelizer"].name == "slic"
    assert [s.name for s in restored.habitat_features] == ["msi", "ith"]


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
        habitat_features=(Spec(name="msi"), Spec(name="ith")),
    )
    assert base.fingerprint() != tweaked.fingerprint()


@pytest.mark.unit
def test_run_policy_defaults_and_validation() -> None:
    """Policy defaults are serial execution with continue-on-failure."""
    policy = RunPolicy()
    assert policy.workers == 1
    assert policy.on_failure == "continue"
    assert policy.backend == "serial"
    with pytest.raises(HABITAPIError):
        RunPolicy(workers=0)
    with pytest.raises(HABITAPIError):
        RunPolicy(on_failure="ignore")
    with pytest.raises(HABITAPIError):
        RunPolicy(backend="dask")


@pytest.mark.unit
def test_run_policy_dict_roundtrip() -> None:
    """Policies tolerate partial payloads and roundtrip fully."""
    policy = RunPolicy.from_dict({"workers": 4, "backend": "process"})
    assert policy.workers == 4
    assert policy.on_failure == "continue"
    assert RunPolicy.from_dict(policy.to_dict()) == policy


@pytest.mark.unit
def test_yaml_roundtrip_for_spec_and_policy(tmp_path) -> None:
    """YAML is a faithful isomorphism for both specification kinds."""
    spec = _habitat_spec()
    spec_path = save_habitat_spec(spec, tmp_path / "spec.yaml")
    assert load_habitat_spec(spec_path) == spec

    policy = RunPolicy(workers=2, seed=42, checkpoint_path=str(tmp_path / "ckpt"))
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
