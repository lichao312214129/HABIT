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

"""Tests for :func:`habit.recipes.run_from_yaml`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

import habit
from habit.exceptions import HABITAPIError
from habit.recipes.result import StudyResult
from habit.recipes.yaml_runner import run_from_yaml

_MODALITY = "t1"
_SUBJECT_IDS = ("subj001", "subj002")


def _synthetic_volumes(seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """Build one subject's (image, mask) pair for a tiny ROI."""
    rng = np.random.default_rng(seed)
    image = np.zeros((6, 10, 10), dtype=np.float32)
    image[:, :, :5] = 100.0
    image[:, :, 5:] = 200.0
    image += rng.normal(0.0, 1.0, size=image.shape).astype(np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[1:5, 2:8, 2:8] = 1
    return image, mask


def _write_nrrd(path: Path, array: np.ndarray) -> None:
    """Write one array as NRRD, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(array), str(path))


def _write_dataset(root: Path) -> Path:
    """Write the two-subject dataset in the v0.1 directory layout."""
    data_root = root / "data"
    for index, subject_id in enumerate(_SUBJECT_IDS):
        image, mask = _synthetic_volumes(seed=100 + index)
        _write_nrrd(data_root / "images" / subject_id / _MODALITY / "image.nrrd", image)
        _write_nrrd(data_root / "masks" / subject_id / _MODALITY / "mask.nrrd", mask)
    return data_root


def _habitat_config_yaml(
    data_dir: Path,
    out_dir: Path,
    *,
    clustering_mode: str = "two_step",
) -> str:
    """Render a minimal v0.1 habitat config for the synthetic dataset."""
    return f"""run_mode: train
data_dir: "{data_dir.as_posix()}"
out_dir: "{out_dir.as_posix()}"
processes: 1
plot_curves: false
save_images: false
save_results_csv: false
habitats_results_format: parquet
random_state: 42
feature_construction:
  voxel_level:
    method: concat(raw({_MODALITY}))
    params: {{}}
  supervoxel_level:
    supervoxel_file_keyword: '*_supervoxel.nrrd'
    method: mean_voxel_features()
    params: {{}}
habitat_segmentation:
  clustering_mode: {clustering_mode}
  supervoxel:
    algorithm: kmeans
    n_clusters: 4
    max_iter: 50
    n_init: 5
    one_step_settings:
      min_clusters: 2
      max_clusters: 4
      fixed_n_clusters: 2
      selection_method: elbow
      plot_validation_curves: false
  habitat:
    algorithm: kmeans
    max_clusters: 4
    habitat_cluster_selection_method:
      - elbow
    fixed_n_clusters: 2
    max_iter: 50
    n_init: 5
"""


def _ml_config_yaml(csv_path: Path, out_dir: Path) -> str:
    """Render a minimal v0.1 ML hold-out config."""
    return f"""run_mode: train
input:
  - path: "{csv_path.as_posix()}"
    subject_id_col: subject
    label_col: label
output: "{out_dir.as_posix()}"
random_state: 0
n_splits: 3
normalization:
  method: z_score
feature_selection_methods: []
models:
  LogisticRegression:
    params:
      max_iter: 500
is_visualize: false
is_save_model: false
"""


@pytest.mark.unit
def test_run_from_yaml_is_public_api() -> None:
    """``run_from_yaml`` resolves from ``import habit`` and the registry."""
    assert habit.run_from_yaml is run_from_yaml
    assert "run_from_yaml" in habit.__all__
    assert "compare_models" in habit.__all__
    assert "pairwise_delong_test" in habit.__all__


@pytest.mark.unit
def test_run_from_yaml_habitat_train_on_synthetic(tmp_path: Path) -> None:
    """Habitat YAML runs through LegacyConfigAdapter and returns StudyResult."""
    data_root = _write_dataset(tmp_path)
    out_dir = tmp_path / "out"
    config_path = tmp_path / "config_habitat_two_step.yaml"
    config_path.write_text(
        _habitat_config_yaml(data_root, out_dir),
        encoding="utf-8",
    )

    result = run_from_yaml(config_path, workflow="habitat")

    assert isinstance(result, StudyResult)
    assert result.habitat_model is not None
    assert len(result.habitat_maps) == len(_SUBJECT_IDS)


@pytest.mark.unit
def test_run_from_yaml_ml_train_on_synthetic_table(tmp_path: Path) -> None:
    """ML hold-out YAML returns a ModelResult without touching disk by default."""
    from habit.datasets.synthetic import make_synthetic_feature_table
    from habit.recipes.modeling import ModelResult

    table = make_synthetic_feature_table(n_rows=24, n_features=6, rng=0)
    csv_path = tmp_path / "features.csv"
    table.frame.to_csv(csv_path, index=False)

    config_path = tmp_path / "config_machine_learning.yaml"
    config_path.write_text(
        _ml_config_yaml(csv_path, tmp_path / "ml_out"),
        encoding="utf-8",
    )

    result = run_from_yaml(config_path, workflow="model")

    assert isinstance(result, ModelResult)
    assert result.pipeline is not None
    assert result.train_metrics


@pytest.mark.unit
def test_run_from_yaml_rejects_unsupported_workflow(tmp_path: Path) -> None:
    """Workflows outside the check-config set raise NotImplementedError."""
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("run_mode: train\n", encoding="utf-8")

    with pytest.raises(NotImplementedError, match="migrate"):
        run_from_yaml(config_path, workflow="migrate")


@pytest.mark.unit
def test_run_from_yaml_preprocess_dispatches_to_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preprocess YAML loads the schema and calls the L4 recipe."""
    from unittest.mock import MagicMock

    out_dir = tmp_path / "out_preprocess"
    config_path = tmp_path / "config_preprocess.yaml"
    config_path.write_text(
        f"""data_dir: "{(out_dir / 'input').as_posix()}"
out_dir: "{out_dir.as_posix()}"
preprocessing:
  resample:
    images: [T1]
    target_spacing: [1.0, 1.0, 1.0]
    img_mode: bilinear
""",
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def _spy(*args: object, **kwargs: object) -> MagicMock:
        calls.append({"args": args, "kwargs": kwargs})
        return MagicMock()

    monkeypatch.setattr("habit.recipes.yaml_runner.preprocess_images", _spy)

    run_from_yaml(config_path, workflow="preprocess")

    assert len(calls) == 1
    assert calls[0]["args"][0].out_dir == str(out_dir)


@pytest.mark.unit
def test_run_from_yaml_requires_workflow_when_path_is_ambiguous(tmp_path: Path) -> None:
    """Ambiguous filenames must pass workflow= explicitly."""
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("run_mode: train\n", encoding="utf-8")

    with pytest.raises(HABITAPIError, match="Cannot guess workflow"):
        run_from_yaml(config_path)


@pytest.mark.unit
def test_run_from_yaml_guesses_habitat_from_path(tmp_path: Path) -> None:
    """Path fragments select habitat without an explicit workflow argument."""
    data_root = _write_dataset(tmp_path)
    out_dir = tmp_path / "out"
    config_path = tmp_path / "habitat" / "config_habitat_two_step.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        _habitat_config_yaml(data_root, out_dir),
        encoding="utf-8",
    )

    result = run_from_yaml(config_path)

    assert isinstance(result, StudyResult)
