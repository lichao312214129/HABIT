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
"""Python API ↔ exported YAML ↔ CLI habitat map parity.

Path A constructs a :class:`~habit.spec.HabitatSpec` in pure Python and runs
:func:`~habit.recipes.two_step`. Path A then saves a complete effective v1
document via :func:`~habit.spec.save_habitat_config`. Path B reloads that
document with :func:`~habit.recipes.run_from_yaml`; path C runs
``habit get-habitat --config`` on the same file. Habitat label maps must be
voxel-identical across A/B/C.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import SimpleITK as sitk
import yaml

from habit import HabitatSpec, RunPolicy, Spec, cohort_from_directory, save_habitat_config
from habit.execution import backend_from_policy
from habit.recipes import run_from_yaml, two_step
from habit.recipes.result import StudyResult
from habit.spec.document import build_habitat_document, load_habitat_config
from habit.utils.subprocess_utils import run_capture_text

_MODALITIES: Tuple[str, ...] = ("pre_contrast", "LAP", "PVP", "delay_3min")
_DEMO_PREPROCESSED = Path("demo_data") / "preprocessed"


def _demo_ready(project_root: Path) -> bool:
    """Return whether the local demo cohort layout is present."""
    root = project_root / _DEMO_PREPROCESSED
    return (root / "images").is_dir() and (root / "masks").is_dir()


def _two_step_demo_spec() -> HabitatSpec:
    """
    Build the quickstart / two-step demo HabitatSpec (seed 42).

    Returns:
        A deterministic two-step specification matching the demo YAML science.
    """
    return HabitatSpec(
        name="habitat_two_step",
        voxel_feature_extractor=Spec(
            "raw",
            {"modalities": list(_MODALITIES)},
        ),
        voxel_feature_preprocessors=(
            Spec(
                "winsorize",
                {"winsor_limits": (0.05, 0.05), "across_features": False},
            ),
            Spec("minmax", {"across_features": False}),
        ),
        supervoxelizer=Spec(
            "kmeans",
            {"n_supervoxels": 50, "max_iter": 300, "n_init": 10},
        ),
        cohort_feature_preprocessors=(
            Spec(
                "binning",
                {
                    "n_bins": 10,
                    "bin_strategy": "uniform",
                    "across_features": False,
                },
            ),
        ),
        habitat_model_fitter=Spec(
            "kmeans",
            {
                "min_habitats": 2,
                "max_habitats": 10,
                "validation": "elbow",
                "max_iter": 300,
                "n_init": 10,
            },
        ),
        habitat_assigner=Spec("nearest_centroid"),
        postprocess_habitat=Spec(
            "connected_components",
            {
                "min_component_size": 100,
                "connectivity": 1,
                "reassign_method": "neighbor_vote",
                "max_iterations": 3,
            },
        ),
        random_seed=42,
    )


def _serial_policy() -> RunPolicy:
    """
    Deterministic serial execution policy (workers=1, no process pool).

    Returns:
        A :class:`RunPolicy` that forces :class:`SerialBackend`.
    """
    return RunPolicy(
        workers=1,
        backend="serial",
        subject_timeout_sec=None,
        subject_spawn_timeout_sec=None,
        resume=False,
    )


def _maps_from_result(result: StudyResult) -> Dict[str, np.ndarray]:
    """Index habitat label arrays by subject id."""
    return {
        habitat_map.subject_id: np.asarray(habitat_map.label_array)
        for habitat_map in result.habitat_maps
    }


def _maps_from_out_dir(out_dir: Path) -> Dict[str, np.ndarray]:
    """Load ``*_habitats.nrrd`` maps written under ``out_dir``."""
    maps: Dict[str, np.ndarray] = {}
    for path in sorted(out_dir.glob("*_habitats.nrrd")):
        subject_id = path.name[: -len("_habitats.nrrd")]
        maps[subject_id] = np.asarray(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))
    return maps


def _assert_maps_equal(
    expected: Dict[str, np.ndarray],
    actual: Dict[str, np.ndarray],
    *,
    label: str,
) -> None:
    """Assert voxel-wise habitat label equality for every subject."""
    assert set(actual) == set(expected), f"{label}: subject id mismatch"
    for subject_id, expected_map in expected.items():
        actual_map = actual[subject_id]
        assert actual_map.shape == expected_map.shape, (
            f"{label}: shape mismatch for {subject_id}"
        )
        mismatched = int(np.sum(actual_map != expected_map))
        assert mismatched == 0, (
            f"{label}: {mismatched} voxels differ for subject {subject_id}"
        )


@pytest.mark.unit
def test_save_habitat_config_expands_defaults(tmp_path: Path) -> None:
    """Exported documents include effective defaults, not only overrides."""
    spec = _two_step_demo_spec()
    document = build_habitat_document(
        spec,
        data_source="demo_data/preprocessed",
        out_dir=tmp_path / "out",
        policy=_serial_policy(),
    )
    assert document["version"] == "1.0"
    assert document["workflow"] == "habitat"
    assert document["spec"]["on_geometry_mismatch"] == "resample_mask"
    assert document["spec"]["postprocess_supervoxel"] is None
    assert document["spec"]["postprocess_habitat"]["name"] == "connected_components"
    assert "workers" in document["policy"]
    assert document["policy"]["backend"] == "serial"
    assert document["output"]["habitats_results_format"] == "parquet"

    yaml_path = save_habitat_config(
        tmp_path / "effective.yaml",
        spec,
        data_source="demo_data/preprocessed",
        out_dir=tmp_path / "out",
        policy=_serial_policy(),
    )
    reloaded = load_habitat_config(yaml_path)
    assert reloaded["spec"]["name"] == "habitat_two_step"
    assert HabitatSpec.from_dict(reloaded["spec"]).fingerprint() == spec.fingerprint()

    # HabitatSpec.to_yaml expands the same effective spec section.
    text = spec.to_yaml(tmp_path / "spec_only.yaml")
    assert "on_geometry_mismatch" in text
    payload = yaml.safe_load(text)
    assert payload["on_geometry_mismatch"] == "resample_mask"


@pytest.mark.integration
def test_python_yaml_cli_habitat_maps_voxel_identical(
    project_root: Path,
    tmp_path: Path,
    cwd_repo_root: None,
) -> None:
    """
    Path A (Python) → save YAML → B (run_from_yaml) and C (CLI) match voxels.
    """
    if not _demo_ready(project_root):
        pytest.skip("demo_data/preprocessed is not present")

    spec = _two_step_demo_spec()
    policy = _serial_policy()
    backend = backend_from_policy(policy)
    data_source = str(_DEMO_PREPROCESSED.as_posix())

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    out_c = tmp_path / "out_c"
    yaml_path = tmp_path / "exported_two_step.yaml"

    # --- Path A: pure Python API ---
    cohort = cohort_from_directory(
        data_source,
        modalities=_MODALITIES,
        roi=_MODALITIES[0],
    )
    result_a = two_step(cohort, spec, backend=backend)
    result_a.save(
        out_a,
        write_maps=True,
        write_units_table=False,
        write_cluster_plots=False,
    )
    maps_a = _maps_from_result(result_a)
    assert maps_a, "path A produced no habitat maps"

    # Export the complete effective document used by B and C.
    save_habitat_config(
        yaml_path,
        spec,
        data_source=data_source,
        out_dir=out_b,
        policy=policy,
        plot_curves=False,
        save_results_csv=False,
    )

    # --- Path B: YAML API (run_from_yaml) ---
    result_b = run_from_yaml(yaml_path, workflow="habitat", save=True)
    assert isinstance(result_b, StudyResult)
    maps_b = _maps_from_result(result_b)
    _assert_maps_equal(maps_a, maps_b, label="A vs B (run_from_yaml)")

    # --- Path C: CLI get-habitat on the same YAML (retarget out_dir) ---
    cli_document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cli_document["output"]["out_dir"] = str(out_c)
    cli_yaml = tmp_path / "exported_two_step_cli.yaml"
    cli_yaml.write_text(
        yaml.safe_dump(cli_document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    completed = run_capture_text(
        [
            sys.executable,
            "-m",
            "habit.cli",
            "get-habitat",
            "--config",
            str(cli_yaml),
        ],
        cwd=str(project_root),
        check=False,
    )
    assert completed.returncode == 0, (
        (completed.stdout or "") + "\n" + (completed.stderr or "")
    )
    maps_c = _maps_from_out_dir(out_c)
    _assert_maps_equal(maps_a, maps_c, label="A vs C (CLI get-habitat)")
