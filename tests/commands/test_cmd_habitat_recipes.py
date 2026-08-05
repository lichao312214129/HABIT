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

"""Phase-4b CLI switch tests: ``get-habitat`` runs on v1 recipes.

These tests prove that ``habit.commands.cmd_habitat.run_habitat`` wires the
v0.1 YAML into the v1 stack (LegacyConfigAdapter -> spec -> cohort -> recipe
-> DirectoryResultWriter) instead of the v0.1 engine, for all three
clustering modes, on a tiny synthetic two-subject dataset. No demo_data, no
golden baseline: everything here finishes in seconds and stays in the
default ``pytest -m "not slow"`` selection.

Coverage:
- train (directory layout and manifest layout) dispatches to the recipe,
  writes the v0.1 artefact layout, and records per-subject results in the
  v1 checkpoint store at the v0.1 location;
- the fitted model is saved as ``habitat_model.habitatmodel``, and predict on
  that archive reproduces the training labels;
- predict on a legacy raw-pickle pipeline is rejected with a v1 migration message;
- a resumed train run skips checkpointed subjects entirely, proven by
  corrupting the input images between runs.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest
import SimpleITK as sitk

import habit.commands.cmd_habitat as cmd_habitat
from habit.commands.cmd_habitat import run_habitat
from habit.contracts.subject import Cohort

#: Modality/ROI key used throughout the synthetic dataset.
_MODALITY = "t1"

#: Subject ids in canonical (sorted) order.
_SUBJECT_IDS = ("subj001", "subj002")


def _synthetic_volumes(seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """
    Build one subject's (image, mask) pair with two intensity regions.

    Args:
        seed: Noise seed, varied per subject.

    Returns:
        ``(image, mask)`` as ``(z, y, x)`` float32/uint8 arrays; the mask
        covers a 4x6x6 box so every subject has exactly 144 ROI voxels.
    """
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
    """
    Write the two-subject dataset in the v0.1 directory layout.

    Args:
        root: Scratch directory.

    Returns:
        The data root holding ``images/`` and ``masks/``.
    """
    data_root = root / "data"
    for index, subject_id in enumerate(_SUBJECT_IDS):
        image, mask = _synthetic_volumes(seed=100 + index)
        _write_nrrd(data_root / "images" / subject_id / _MODALITY / "image.nrrd", image)
        _write_nrrd(data_root / "masks" / subject_id / _MODALITY / "mask.nrrd", mask)
    return data_root


def _write_manifest(root: Path, data_root: Path) -> Path:
    """
    Write a v0.1 ``file_*.yaml`` manifest equivalent of the directory layout.

    Args:
        root: Scratch directory the manifest lives in (relative paths
            resolve against it, v0.1 semantics).
        data_root: Dataset root written by :func:`_write_dataset`.

    Returns:
        The manifest path.
    """
    lines = ["auto_select_first_file: true", "images:"]
    for subject_id in _SUBJECT_IDS:
        rel = (
            data_root / "images" / subject_id / _MODALITY / "image.nrrd"
        ).relative_to(root)
        lines.append(f"  {subject_id}:")
        lines.append(f"    {_MODALITY}: {rel.as_posix()}")
    lines.append("masks:")
    for subject_id in _SUBJECT_IDS:
        rel = (
            data_root / "masks" / subject_id / _MODALITY / "mask.nrrd"
        ).relative_to(root)
        lines.append(f"  {subject_id}:")
        lines.append(f"    {_MODALITY}: {rel.as_posix()}")
    manifest = root / "file_synthetic.yaml"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _config_yaml(
    data_dir: Path,
    out_dir: Path,
    *,
    clustering_mode: str = "two_step",
    run_mode: str = "train",
    pipeline_path: Optional[Path] = None,
) -> str:
    """
    Render a minimal v0.1 habitat config for the synthetic dataset.

    Args:
        data_dir: Directory layout root or manifest YAML.
        out_dir: Output directory.
        clustering_mode: One of ``two_step``/``one_step``/``direct_pooling``.
        run_mode: ``train`` or ``predict``.
        pipeline_path: Fitted pipeline path (predict mode only).

    Returns:
        YAML text.
    """
    pipeline_lines = ""
    if pipeline_path is not None:
        pipeline_lines = f'pipeline_path: "{pipeline_path.as_posix()}"\n'
    return f"""run_mode: {run_mode}
data_dir: "{data_dir.as_posix()}"
out_dir: "{out_dir.as_posix()}"
{pipeline_lines}processes: 1
plot_curves: false
save_images: true
save_results_csv: true
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


def _write_config(root: Path, content: str, name: str = "config.yaml") -> Path:
    """Write one config file under ``root`` and return its path."""
    path = root / name
    path.write_text(content, encoding="utf-8")
    return path


def _read_labels(path: Path) -> np.ndarray:
    """Read a label map back as a NumPy array."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


class _RecipeSpy:
    """
    Wrapper recording recipe invocations while delegating to the real one.

    Attributes:
        calls: Positional arguments of every invocation (cohort first).
    """

    def __init__(self, recipe: Callable[..., Any]) -> None:
        self._recipe = recipe
        self.calls: List["tuple[Any, ...]"] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record the call and run the wrapped recipe."""
        self.calls.append(args)
        return self._recipe(*args, **kwargs)


def _spy_on_recipe(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> _RecipeSpy:
    """Replace one entry of the CLI's recipe table with a recording spy."""
    spy = _RecipeSpy(cmd_habitat._RECIPE_BY_MODE[mode])
    monkeypatch.setitem(cmd_habitat._RECIPE_BY_MODE, mode, spy)
    return spy


@pytest.mark.cli
def test_train_two_step_dispatches_to_recipe_and_writes_v0_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two-step train runs the recipe, not the v0.1 engine, and writes v0.1 artefacts."""
    data_root = _write_dataset(tmp_path)
    out_dir = tmp_path / "out_two_step"
    config_path = _write_config(tmp_path, _config_yaml(data_root, out_dir))
    spy = _spy_on_recipe(monkeypatch, "two_step")

    run_habitat(str(config_path), debug_mode=False, mode=None, pipeline_path=None, exit_on_error=False)

    # The recipe ran exactly once, on a two-subject cohort in sorted order.
    assert len(spy.calls) == 1
    cohort = spy.calls[0][0]
    assert isinstance(cohort, Cohort)
    assert cohort.subject_ids == _SUBJECT_IDS

    # v0.1 artefact layout: maps, supervoxel maps, table, manifests, model.
    for subject_id in _SUBJECT_IDS:
        assert (out_dir / f"{subject_id}_habitats.nrrd").is_file()
        assert (out_dir / f"{subject_id}_supervoxel.nrrd").is_file()
        labels = _read_labels(out_dir / f"{subject_id}_habitats.nrrd")
        assert set(np.unique(labels)) == {0, 1, 2}
    assert (out_dir / "habitats.parquet").is_file()
    assert (out_dir / "habitat_model.habitatmodel").is_file()
    assert (out_dir / "run_manifest.json").is_file()

    # Stage-5 checkpoint strategy: the v1 store sits at the v0.1 location
    # with one entry per subject per cached stage (units + labels), in the
    # v1 digest format -- never the v0.1 manifest/subjects layout.
    checkpoint_dir = out_dir / ".habitat_checkpoint"
    assert checkpoint_dir.is_dir()
    assert len(list(checkpoint_dir.glob("*.pkl"))) == 2 * len(_SUBJECT_IDS)
    assert not (checkpoint_dir / "manifest.json").exists()
    assert not (checkpoint_dir / "subjects").exists()

    # save_images: true also enables the v1 population clustering scatter.
    assert (
        out_dir / "visualizations" / "habitat_clustering" / "habitat_clustering_2D.png"
    ).is_file()


@pytest.mark.cli
def test_train_two_step_via_manifest_builds_cohort_in_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest input is parsed at L5 and yields the same cohort shape."""
    data_root = _write_dataset(tmp_path)
    manifest = _write_manifest(tmp_path, data_root)
    out_dir = tmp_path / "out_manifest"
    config_path = _write_config(tmp_path, _config_yaml(manifest, out_dir))
    spy = _spy_on_recipe(monkeypatch, "two_step")

    run_habitat(str(config_path), debug_mode=False, mode=None, pipeline_path=None, exit_on_error=False)

    assert len(spy.calls) == 1
    cohort = spy.calls[0][0]
    assert cohort.subject_ids == _SUBJECT_IDS
    subject = cohort[0]
    assert sorted(subject.images) == [_MODALITY]
    assert sorted(subject.masks) == [_MODALITY]
    assert (out_dir / "habitats.parquet").is_file()


@pytest.mark.cli
@pytest.mark.parametrize("mode", ["one_step", "direct_pooling"])
def test_train_other_modes_dispatch_to_their_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """one_step and direct_pooling also run through their v1 recipes."""
    data_root = _write_dataset(tmp_path)
    out_dir = tmp_path / f"out_{mode}"
    config_path = _write_config(
        tmp_path, _config_yaml(data_root, out_dir, clustering_mode=mode)
    )
    spy = _spy_on_recipe(monkeypatch, mode)

    run_habitat(str(config_path), debug_mode=False, mode=None, pipeline_path=None, exit_on_error=False)

    assert len(spy.calls) == 1
    for subject_id in _SUBJECT_IDS:
        assert (out_dir / f"{subject_id}_habitats.nrrd").is_file()
    assert (out_dir / "habitats.parquet").is_file()
    # one_step fits per-subject models: no cohort model artefact (decision 5).
    if mode == "one_step":
        assert not (out_dir / "habitat_model.habitatmodel").exists()
    else:
        assert (out_dir / "habitat_model.habitatmodel").is_file()


@pytest.mark.cli
def test_predict_with_v1_archive_relabels_training_data_identically(
    tmp_path: Path,
) -> None:
    """Predict through the v1 archive reproduces the training label maps."""
    data_root = _write_dataset(tmp_path)
    train_out = tmp_path / "out_train"
    train_config = _write_config(tmp_path, _config_yaml(data_root, train_out))
    run_habitat(str(train_config), debug_mode=False, mode=None, pipeline_path=None, exit_on_error=False)

    predict_out = tmp_path / "out_predict"
    predict_config = _write_config(
        tmp_path,
        _config_yaml(
            data_root,
            predict_out,
            run_mode="predict",
            pipeline_path=train_out / "habitat_model.habitatmodel",
        ),
        name="config_predict.yaml",
    )
    run_habitat(str(predict_config), debug_mode=False, mode=None, pipeline_path=None, exit_on_error=False)

    for subject_id in _SUBJECT_IDS:
        expected = _read_labels(train_out / f"{subject_id}_habitats.nrrd")
        actual = _read_labels(predict_out / f"{subject_id}_habitats.nrrd")
        assert np.array_equal(expected, actual), subject_id
    assert (predict_out / "habitats.parquet").is_file()


@pytest.mark.cli
def test_predict_with_legacy_pickle_rejects_v0_artefact(
    tmp_path: Path,
) -> None:
    """A raw-pickle pipeline is rejected with a v1 migration message."""
    data_root = _write_dataset(tmp_path)
    legacy_pkl = tmp_path / "habitat_pipeline.pkl"
    with legacy_pkl.open("wb") as handle:
        pickle.dump({"legacy": "pipeline"}, handle)
    out_dir = tmp_path / "out_legacy_predict"
    config_path = _write_config(
        tmp_path,
        f"""run_mode: predict
data_dir: "{data_root.as_posix()}"
out_dir: "{out_dir.as_posix()}"
pipeline_path: "{legacy_pkl.as_posix()}"
habitat_segmentation:
  clustering_mode: two_step
""",
    )

    with pytest.raises(ValueError, match="Legacy v0.1 pickle"):
        run_habitat(
            str(config_path),
            debug_mode=False,
            mode=None,
            pipeline_path=None,
            exit_on_error=False,
        )


@pytest.mark.cli
def test_save_result_forwards_save_images_to_cluster_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_save_result passes write_cluster_plots from the v0.1 save_images switch."""
    data_root = _write_dataset(tmp_path)
    out_dir = tmp_path / "out_save_flags"
    config_path = _write_config(
        tmp_path,
        _config_yaml(data_root, out_dir).replace("save_images: true", "save_images: false"),
    )
    save_calls: List[Dict[str, Any]] = []

    def _spy_save(self: Any, out: Any, **kwargs: Any) -> Path:
        save_calls.append({"out": out, **kwargs})
        return Path(out)

    monkeypatch.setattr("habit.recipes.result.StudyResult.save", _spy_save)

    run_habitat(str(config_path), debug_mode=False, mode=None, pipeline_path=None, exit_on_error=False)

    assert len(save_calls) == 1
    assert save_calls[0]["write_maps"] is False
    assert save_calls[0]["write_cluster_plots"] is False


@pytest.mark.cli
def test_resume_second_run_skips_checkpointed_subjects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A resumed run restores every subject from the v1 checkpoint store.

    Corrupting every input file between runs is the proof: cohort assembly
    only builds lazy references, so any per-subject cache miss would reach
    the unreadable pixels and fail the run. Completing the second run with
    identical labels can only mean every stage was served from the store.
    """
    data_root = _write_dataset(tmp_path)
    out_dir = tmp_path / "out_resume"
    config_path = _write_config(tmp_path, _config_yaml(data_root, out_dir))

    run_habitat(
        str(config_path),
        debug_mode=False,
        mode=None,
        pipeline_path=None,
        resume=True,
        exit_on_error=False,
    )
    assert (out_dir / ".habitat_checkpoint").is_dir()
    expected = {
        subject_id: _read_labels(out_dir / f"{subject_id}_habitats.nrrd")
        for subject_id in _SUBJECT_IDS
    }

    for nrrd in data_root.rglob("*.nrrd"):
        nrrd.write_bytes(b"corrupted: not a NRRD file")

    run_habitat(
        str(config_path),
        debug_mode=False,
        mode=None,
        pipeline_path=None,
        resume=True,
        exit_on_error=False,
    )

    for subject_id in _SUBJECT_IDS:
        actual = _read_labels(out_dir / f"{subject_id}_habitats.nrrd")
        assert np.array_equal(expected[subject_id], actual), subject_id

    echoed = capsys.readouterr().out
    assert "Resume: enabled" in echoed
