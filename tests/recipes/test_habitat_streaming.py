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
"""Streaming persistence for the one-step habitat design.

With ``writer=`` / ``retain=`` / ``on_subject_complete=`` a one-step fit
persists each subject's artefacts the moment the backend yields it, and the
in-memory result keeps only small tables:

* habitat maps (and per-subject ``.habitatmodel`` files) land on disk
  per subject, so a crashed run keeps completed subjects;
* voxel-level clustering units -- the memory-dominant payload -- are
  aggregated into units-table rows inside the workers and never cross the
  process boundary;
* the units table written by ``StudyResult.save`` is bit-identical to the
  one a default in-memory run produces.

Everything below runs on the in-memory synthetic cohort and finishes in
seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from habit.adapters.writers import DirectoryResultWriter
from habit.contracts.habitat import HabitatMap, HabitatModel
from habit.contracts.subject import Cohort, Subject
from habit.datasets import make_synthetic_cohort
from habit.exceptions import HABITAPIError
from habit.execution.checkpoint import CheckpointStore
from habit.recipes.study import Study
from habit.spec.specs import HabitatSpec, Spec

#: Volume shape for every cohort in this module; small enough that kmeans
#: is instantaneous, large enough for three z-band habitats.
_SHAPE = (16, 16, 16)


def _one_step_spec(n_habitats: int = 3) -> HabitatSpec:
    """
    Build a fast one-step habitat spec with a fixed habitat count.

    Args:
        n_habitats: Fixed cluster count for the kmeans fitter.

    Returns:
        A fully seeded :class:`~habit.spec.specs.HabitatSpec` without a
        supervoxel stage.
    """
    return HabitatSpec(
        name="streaming_test",
        voxel_feature_extractor=Spec(
            name="raw",
            params={"modalities": ["T1", "T2"], "roi": "tumor"},
        ),
        supervoxelizer=None,
        habitat_model_fitter=Spec(
            name="kmeans",
            params={"n_habitats": n_habitats, "n_init": 2},
        ),
        habitat_assigner=Spec(name="nearest_centroid", params={}),
        habitat_features=(Spec(name="volume"),),
        random_seed=0,
    )


def _two_step_spec() -> HabitatSpec:
    """Build the two-step variant of the test spec (SLIC supervoxels)."""
    return HabitatSpec(
        name="streaming_test",
        voxel_feature_extractor=Spec(
            name="raw",
            params={"modalities": ["T1", "T2"], "roi": "tumor"},
        ),
        supervoxelizer=Spec(
            name="slic", params={"n_supervoxels": 8, "compactness": 5.0}
        ),
        habitat_model_fitter=Spec(
            name="kmeans",
            params={"n_habitats": 3, "n_init": 2},
        ),
        habitat_assigner=Spec(name="nearest_centroid", params={}),
        habitat_features=(Spec(name="volume"),),
        random_seed=0,
    )


def _cohort(n_subjects: int = 3) -> Cohort:
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

    Any cache miss would fail inside the pipeline (there is no ROI to
    read), so a completed sabotaged run proves every subject came back
    from the checkpoint.
    """
    return Cohort(
        [
            Subject(subject_id=subject.subject_id, images={}, masks={})
            for subject in cohort
        ],
        name=cohort.name,
    )


def _read_map_arrays(directory: Path) -> dict:
    """Read every ``*_habitats.nrrd`` under ``directory`` into arrays."""
    import SimpleITK as sitk

    arrays = {}
    for path in sorted(directory.glob("*_habitats.nrrd")):
        arrays[path.name] = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return arrays


@pytest.mark.unit
def test_streaming_matches_batch_run_and_bounds_memory(tmp_path: Path) -> None:
    """Streaming persists per subject and reproduces the batch artefacts."""
    cohort = _cohort(3)

    # Baseline: default in-memory run, saved in one batch at the end.
    batch_dir = tmp_path / "batch"
    batch = Study(spec=_one_step_spec(), design="one_step").fit_predict(cohort)
    batch.save(batch_dir, table_format="parquet", write_cluster_plots=False)

    # Streaming run: maps + per-subject models land on disk as subjects
    # complete; the callback records completion order.
    stream_dir = tmp_path / "stream"
    completed: List[Tuple[str, HabitatMap, HabitatModel]] = []
    writer = DirectoryResultWriter(stream_dir)
    streamed = Study(spec=_one_step_spec(), design="one_step").fit_predict(
        cohort,
        writer=writer,
        retain="tables",
        on_subject_complete=lambda subject, habitat_map, model: completed.append(
            (str(subject.subject_id), habitat_map, model)
        ),
    )
    streamed.save(stream_dir, table_format="parquet", write_cluster_plots=False)

    # Per-subject artefacts were written during the fit.
    expected_ids = {str(subject.subject_id) for subject in cohort}
    assert {path.stem[: -len("_habitats")] for path in stream_dir.glob("*_habitats.nrrd")} == expected_ids
    assert {path.name[: -len(".habitatmodel")] for path in stream_dir.glob("*.habitatmodel")} == expected_ids
    assert {entry[0] for entry in completed} == expected_ids

    # The in-memory result is bounded: no maps, no voxel-level units.
    assert streamed.habitat_maps == ()
    assert streamed.units == ()
    assert len(streamed.units_rows) == len(cohort)
    assert streamed.maps_persisted
    assert set(streamed.subject_models) == set(batch.subject_models)

    # Cohort-level artefacts match the batch run exactly.
    pd.testing.assert_frame_equal(
        pd.read_parquet(batch_dir / "habitats.parquet"),
        pd.read_parquet(stream_dir / "habitats.parquet"),
    )
    pd.testing.assert_frame_equal(batch.features.frame, streamed.features.frame)

    # Streamed maps are voxel-identical to the batch-written ones.
    batch_arrays = _read_map_arrays(batch_dir)
    stream_arrays = _read_map_arrays(stream_dir)
    assert set(batch_arrays) == set(stream_arrays)
    for name in batch_arrays:
        assert np.array_equal(batch_arrays[name], stream_arrays[name])


@pytest.mark.unit
def test_streaming_tables_requires_writer(tmp_path: Path) -> None:
    """retain='tables' without a writer would lose the maps: fail fast."""
    with pytest.raises(HABITAPIError, match="retain='tables'"):
        Study(spec=_one_step_spec(), design="one_step").fit_predict(
            _cohort(2), retain="tables"
        )


@pytest.mark.unit
def test_streaming_rejects_unknown_retain_mode(tmp_path: Path) -> None:
    """An unknown retain mode is rejected before any compute."""
    with pytest.raises(HABITAPIError, match="Unknown Report.retain"):
        Study(spec=_one_step_spec(), design="one_step").fit_predict(
            _cohort(2), retain="nothing"
        )


@pytest.mark.unit
def test_streaming_rejected_for_cohort_level_designs(tmp_path: Path) -> None:
    """two_step / direct_pooling do not stream yet: explicit error."""
    writer = DirectoryResultWriter(tmp_path / "out")
    with pytest.raises(HABITAPIError, match="one_step design only"):
        Study(spec=_two_step_spec(), design="two_step").fit_predict(
            _cohort(2), writer=writer
        )
    with pytest.raises(HABITAPIError, match="one_step design only"):
        Study(spec=_one_step_spec(), design="direct_pooling").fit_predict(
            _cohort(2), retain="maps"
        )


@pytest.mark.unit
def test_streaming_resume_rewrites_maps_from_checkpoint(tmp_path: Path) -> None:
    """A resumed run re-persists streamed artefacts from cached payloads."""
    cohort = _cohort(3)
    store = CheckpointStore(tmp_path / "ckpt")
    out_dir = tmp_path / "out"

    first = Study(spec=_one_step_spec(), design="one_step").fit_predict(
        cohort,
        checkpoint=store,
        writer=DirectoryResultWriter(out_dir),
        retain="tables",
    )
    assert len(store) == 3
    assert len(list(out_dir.glob("*_habitats.nrrd"))) == 3

    # Simulate a crash that lost the streamed maps (checkpoint survived).
    for path in out_dir.glob("*_habitats.nrrd"):
        path.unlink()
    assert list(out_dir.glob("*_habitats.nrrd")) == []

    second = Study(spec=_one_step_spec(), design="one_step").fit_predict(
        _sabotaged_cohort(cohort),
        checkpoint=store,
        writer=DirectoryResultWriter(out_dir),
        retain="tables",
    )

    # Nothing recomputed (sabotaged cohort would fail on a miss), and the
    # hook re-persisted every map from the cached slim payloads.
    assert len(store) == 3
    assert len(list(out_dir.glob("*_habitats.nrrd"))) == 3
    assert set(second.subject_models) == set(first.subject_models)
    pd.testing.assert_frame_equal(
        pd.concat(list(first.units_rows), ignore_index=True),
        pd.concat(list(second.units_rows), ignore_index=True),
    )


@pytest.mark.unit
def test_retain_maps_keeps_maps_but_drops_units(tmp_path: Path) -> None:
    """retain='maps': maps stay in memory AND on disk; units do not."""
    out_dir = tmp_path / "out"
    result = Study(spec=_one_step_spec(), design="one_step").fit_predict(
        _cohort(2),
        writer=DirectoryResultWriter(out_dir),
        retain="maps",
    )
    assert len(result.habitat_maps) == 2
    assert result.units == ()
    assert len(result.units_rows) == 2
    assert result.maps_persisted
    assert len(list(out_dir.glob("*_habitats.nrrd"))) == 2

    # The units table still comes out, sourced from the aggregated rows.
    result.save(out_dir, table_format="csv", write_cluster_plots=False)
    table = pd.read_csv(out_dir / "habitats.csv")
    assert set(table["subject"]) == {
        str(subject.subject_id) for subject in _cohort(2)
    }


@pytest.mark.unit
def test_default_run_keeps_everything_in_memory(tmp_path: Path) -> None:
    """Without streaming options the historical behaviour is unchanged."""
    result = Study(spec=_one_step_spec(), design="one_step").fit_predict(_cohort(2))
    assert len(result.habitat_maps) == 2
    assert len(result.units) == 2
    assert result.units_rows == ()
    assert not result.maps_persisted

    out_dir = tmp_path / "out"
    result.save(out_dir, table_format="parquet", write_cluster_plots=False)
    assert len(list(out_dir.glob("*_habitats.nrrd"))) == 2
    assert (out_dir / "habitats.parquet").exists()


@pytest.mark.unit
def test_report_api_persists_and_draws_per_subject(tmp_path: Path) -> None:
    """Study.fit_predict(report=) streams maps, models, and figure atoms."""
    pytest.importorskip("matplotlib")
    from habit.report import Overlay, Report, VolumeFractions

    out_dir = tmp_path / "out"
    writer = DirectoryResultWriter(out_dir)
    report = Report(
        persist=("habitat_map", "subject_model"),
        retain="tables",
        figures=(Overlay(modality="T1"), VolumeFractions()),
        writer=writer,
    )
    cohort = _cohort(2)
    result = Study(spec=_one_step_spec(), design="one_step").fit_predict(
        cohort, report=report
    )

    assert result.habitat_maps == ()
    assert result.units == ()
    assert result.maps_persisted
    expected = {str(subject.subject_id) for subject in cohort}
    assert {path.stem[: -len("_habitats")] for path in out_dir.glob("*_habitats.nrrd")} == expected
    assert {path.stem for path in out_dir.glob("*.habitatmodel")} == expected
    fig_dir = out_dir / "figures"
    overlays = {path.name for path in fig_dir.glob("*_T1_overlay.png")}
    volumes = {path.name for path in fig_dir.glob("*_volume_fractions.png")}
    assert overlays == {f"{sid}_T1_overlay.png" for sid in expected}
    assert volumes == {f"{sid}_volume_fractions.png" for sid in expected}


@pytest.mark.unit
def test_report_figures_require_a_destination(tmp_path: Path) -> None:
    """Figures without figure_dir or a rooted writer fail before compute."""
    from habit.report import Overlay, Report

    report = Report(figures=(Overlay(modality="T1"),))
    with pytest.raises(HABITAPIError, match="figure_dir"):
        Study(spec=_one_step_spec(), design="one_step").fit_predict(
            _cohort(2), report=report
        )


@pytest.mark.unit
def test_report_rejects_unknown_figure_layout() -> None:
    """figure_layout is a closed set: flat or by_subject."""
    from habit.report import Report

    with pytest.raises(HABITAPIError, match="figure_layout"):
        Report(figure_layout="by_type")


@pytest.mark.unit
def test_report_resolve_figure_path_layouts(tmp_path: Path) -> None:
    """flat keeps the stem; by_subject nests under the subject id."""
    from habit.report import Report

    fig_dir = tmp_path / "figures"
    flat = Report(figure_dir=fig_dir, figure_layout="flat")
    nested = Report(figure_dir=fig_dir, figure_layout="by_subject")
    stem = "sub1_T1_overlay"
    assert flat.resolve_figure_path("sub1", stem) == fig_dir / "sub1_T1_overlay.png"
    assert nested.resolve_figure_path("sub1", stem) == fig_dir / "sub1" / "T1_overlay.png"
    assert nested.resolve_figure_path("sub1", "custom") == fig_dir / "sub1" / "custom.png"


@pytest.mark.unit
def test_report_graph_atoms_stems_and_by_subject_paths(tmp_path: Path) -> None:
    """2D graph atoms use graph_slice / graph_network_2d stems."""
    from habit.kernels.habitat_graph import HabitatGraphFeatureOptions
    from habit.report import GraphNetwork2D, GraphSlice, Report

    options = HabitatGraphFeatureOptions(edge_method="min_distance", block_size=8)
    slice_atom = GraphSlice(options=options)
    network_atom = GraphNetwork2D(options=options)
    assert slice_atom.stem("sub1") == "sub1_graph_slice"
    assert network_atom.stem("sub1") == "sub1_graph_network_2d"
    report = Report(figure_dir=tmp_path / "figures", figure_layout="by_subject")
    assert report.resolve_figure_path("sub1", slice_atom.stem("sub1")) == (
        tmp_path / "figures" / "sub1" / "graph_slice.png"
    )
    assert report.resolve_figure_path("sub1", network_atom.stem("sub1")) == (
        tmp_path / "figures" / "sub1" / "graph_network_2d.png"
    )


@pytest.mark.unit
def test_report_graph_atoms_draw_2d_figures() -> None:
    """GraphSlice and GraphNetwork2D return matplotlib figures for a map."""
    pytest.importorskip("matplotlib")
    from habit.kernels.habitat_graph import HabitatGraphFeatureOptions
    from habit.report import GraphNetwork2D, GraphSlice
    from habit.report.api import SubjectContext

    labels = np.zeros((32, 32), dtype=np.int32)
    labels[4:12, 4:12] = 1
    labels[16:26, 6:14] = 2

    class _Map:
        label_array = labels
        habitat_ids = (1, 2)

    class _Subject:
        subject_id = "sub1"

    ctx = SubjectContext(
        subject=_Subject(),  # type: ignore[arg-type]
        habitat_map=_Map(),  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
    )
    options = HabitatGraphFeatureOptions(edge_method="min_distance", block_size=8)
    slice_fig = GraphSlice(options=options).draw(ctx)
    network_fig = GraphNetwork2D(options=options).draw(ctx)
    assert slice_fig is not None
    assert network_fig is not None
    import matplotlib.pyplot as plt

    plt.close(slice_fig)
    plt.close(network_fig)
