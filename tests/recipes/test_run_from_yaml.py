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
def test_run_from_yaml_registered_single_roi_manifest_nii_gz(tmp_path: Path) -> None:
    """
    Direct ``.nii.gz`` paths in a manifest with ``auto_select_first_file: false``
    (already-registered multi-modal + one shared ROI) load and train.
    """
    import yaml

    data_root = tmp_path / "registered"
    for index, subject_id in enumerate(_SUBJECT_IDS):
        image, mask = _synthetic_volumes(seed=200 + index)
        subject_dir = data_root / subject_id
        subject_dir.mkdir(parents=True)
        # Two modalities share one ROI file (registered single-mask layout).
        sitk.WriteImage(
            sitk.GetImageFromArray(image), str(subject_dir / "T1.nii.gz")
        )
        sitk.WriteImage(
            sitk.GetImageFromArray(image + 10.0), str(subject_dir / "T2.nii.gz")
        )
        sitk.WriteImage(
            sitk.GetImageFromArray(mask), str(subject_dir / "roi.nii.gz")
        )

    manifest = {
        "auto_select_first_file": False,
        "images": {
            sid: {
                "T1": str((data_root / sid / "T1.nii.gz").as_posix()),
                "T2": str((data_root / sid / "T2.nii.gz").as_posix()),
            }
            for sid in _SUBJECT_IDS
        },
        "masks": {
            sid: {"T1": str((data_root / sid / "roi.nii.gz").as_posix())}
            for sid in _SUBJECT_IDS
        },
    }
    manifest_path = tmp_path / "file_registered_single_roi.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    out_dir = tmp_path / "out"
    config_path = tmp_path / "config_habitat.yaml"
    config_path.write_text(
        f"""run_mode: train
data_dir: "{manifest_path.as_posix()}"
out_dir: "{out_dir.as_posix()}"
processes: 1
plot_curves: false
save_images: true
save_results_csv: true
habitats_results_format: csv
random_state: 42
feature_construction:
  voxel_level:
    method: concat(raw(T1), raw(T2))
    params: {{}}
  supervoxel_level:
    supervoxel_file_keyword: '*_supervoxel.nrrd'
    method: mean_voxel_features()
    params: {{}}
habitat_segmentation:
  clustering_mode: two_step
  supervoxel:
    algorithm: kmeans
    n_clusters: 4
    max_iter: 50
    n_init: 5
  individual_level:
    algorithm: kmeans
    max_clusters: 3
    habitat_cluster_selection_method:
      - silhouette
    max_iter: 50
    n_init: 5
  group_level:
    algorithm: kmeans
    max_clusters: 3
    habitat_cluster_selection_method:
      - silhouette
    fixed_n_clusters: 2
    max_iter: 50
    n_init: 5
""",
        encoding="utf-8",
    )

    result = run_from_yaml(config_path, workflow="habitat", save=True)

    assert isinstance(result, StudyResult)
    assert result.habitat_model is not None
    assert len(result.habitat_maps) == len(_SUBJECT_IDS)
    assert (out_dir / f"{_SUBJECT_IDS[0]}_habitats.nrrd").is_file()


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


def _ml_v1_steps_document(csv_path: Path, out_dir: Path) -> str:
    """
    Render a native v1 ML document declaring an interleaved ``steps`` chain.

    The order ``zscore -> variance -> minmax`` is the point: a preprocessor,
    a selector, then a SECOND preprocessor is unrepresentable in the
    deprecated three-slot layout, which offers a selector only the two
    positions "before all preprocessing" and "after all of it".

    Args:
        csv_path: Feature table written by the caller.
        out_dir: Output directory (unused unless ``save=True``).

    Returns:
        str: The YAML document text.
    """
    return f"""version: '1.0'
workflow: model
mode: train
spec:
  name: ml_model
  steps:
    - name: zscore
      params: {{}}
    - name: variance
      params:
        threshold: 0.0
    - name: minmax
      params: {{}}
  classifier:
    name: LogisticRegression
    params:
      max_iter: 500
  metrics: []
  random_seed: 0
data:
  input:
    - path: "{csv_path.as_posix()}"
      subject_id_col: subject
      label_col: label
output:
  out_dir: "{out_dir.as_posix()}"
  is_save_model: false
"""


@pytest.mark.unit
def test_run_from_yaml_v1_ml_document_with_ordered_steps(tmp_path: Path) -> None:
    """A v1 ML document declaring ``spec.steps`` runs end to end."""
    from habit.datasets.synthetic import make_synthetic_feature_table
    from habit.recipes.modeling import ModelResult

    table = make_synthetic_feature_table(n_rows=24, n_features=6, rng=0)
    csv_path = tmp_path / "features.csv"
    table.frame.to_csv(csv_path, index=False)

    config_path = tmp_path / "config_machine_learning_steps_v1.yaml"
    config_path.write_text(
        _ml_v1_steps_document(csv_path, tmp_path / "ml_out"),
        encoding="utf-8",
    )

    result = run_from_yaml(config_path, workflow="model")

    assert isinstance(result, ModelResult)
    assert [
        component.spec.name for component in result.pipeline.components
    ] == ["zscore", "variance", "minmax"]
    # The ordered layout reaches the manifest, so the run is traceable back
    # to the document that produced it.
    assert "steps" in result.manifest.spec_payload
    assert [
        entry["name"] for entry in result.manifest.spec_payload["steps"]
    ] == ["zscore", "variance", "minmax"]


@pytest.mark.unit
def test_shipped_steps_example_declares_an_interleaved_pipeline() -> None:
    """
    The example config under ``config/`` assembles the pipeline it advertises.

    A demo config that no test builds is a demo config that rots; this one
    needs no data, so the assembly is checkable even where ``demo_data/`` is
    absent.
    """
    import yaml

    from habit.domain.assembly import build_table_pipeline
    from habit.spec.legacy import validate_v1_document
    from habit.spec.specs import MLSpec

    config_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "machine_learning"
        / "config_machine_learning_steps_v1.yaml"
    )
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_v1_document(document, workflow="cv")
    spec = MLSpec.from_dict(document["spec"])
    pipeline = build_table_pipeline(spec)
    assert [component.spec.name for component in pipeline.components] == [
        "variance",
        "zscore",
        "correlation",
        "minmax",
        "lasso",
    ]


@pytest.mark.unit
def test_v1_reverse_translation_reads_before_z_score_off_the_step_position() -> None:
    """
    ``steps`` positions map back onto v0's single ``before_z_score`` boolean.

    v0 had no ordered list, only that flag; a selector is
    ``before_z_score`` exactly when no preprocessor precedes it. The v0
    payload is what loads the feature table, so it has to stay derivable
    from either layout.
    """
    from habit.recipes.yaml_runner import _v0_selection_methods_from_spec

    methods = _v0_selection_methods_from_spec(
        {
            "steps": [
                {"name": "variance", "params": {"top_k": 5}},
                {"name": "zscore", "params": {}},
                {"name": "correlation", "params": {"threshold": 0.9}},
            ]
        }
    )
    assert methods == [
        {"method": "variance", "params": {"top_k": 5, "before_z_score": True}},
        {"method": "correlation", "params": {"threshold": 0.9}},
    ]


@pytest.mark.unit
def test_v1_reverse_translation_still_reads_the_deprecated_chains() -> None:
    """The deprecated layout keeps its verbatim reverse translation."""
    from habit.recipes.yaml_runner import _v0_selection_methods_from_spec

    methods = _v0_selection_methods_from_spec(
        {
            "pre_preprocessing_feature_selectors": [
                {"name": "variance", "params": {"top_k": 5}}
            ],
            "feature_selectors": [{"name": "correlation", "params": {}}],
        }
    )
    assert methods == [
        {"method": "variance", "params": {"top_k": 5, "before_z_score": True}},
        {"method": "correlation", "params": {}},
    ]


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
