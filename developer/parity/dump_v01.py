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
"""Dump layered parity checkpoints from the FROZEN v0.1 worktree.

This script is read-only with respect to the v0.1 library: it drives the
existing pipeline steps in their configured order and snapshots the
per-subject payload between them. Nothing in ``habit.core`` is patched or
monkey-patched, so the numbers recorded here are the numbers the v0.1 CLI
would have produced for the same YAML.

The v0.1 library must be first on ``sys.path``::

    $env:PYTHONPATH = "F:\\work\\habit_v01_parity"
    python developer/parity/dump_v01.py --config <yaml> --leg-dir <dir>

Determinism is enforced before numpy loads (single-threaded BLAS) and by
the YAML itself (processes=1, resume off, fixed seed / n_init / k).
"""

from __future__ import annotations

import os

# Must precede any numpy / sklearn import so the BLAS backends see them.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _parity_io as pio  # noqa: E402

# Frozen v0.1 worktree. Prefer HABIT_V01_ROOT; fall back to the sibling
# checkout used by the Phase-1 / Phase-2 harness on this machine.
_V01_ROOT = Path(os.environ.get("HABIT_V01_ROOT", r"F:\work\habit_v01_parity")).resolve()


def _force_v01_on_path() -> None:
    """
    Ensure the frozen v0.1 tree is the first importable ``habit``.

    A ``sitecustomize`` / ``.pth`` may put the v1 working tree at
    ``sys.path[0]`` so CLI tools always see the workspace; that would
    silently make this leg dump v1 numbers. Strip those entries and put
    the frozen tree first before any ``import habit``.
    """
    blocked = {"habit_project", "habit_project_v1"}
    cleaned: List[str] = []
    for entry in sys.path:
        try:
            name = Path(entry).resolve().name
        except OSError:
            name = entry
        if name in blocked:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned
    root = str(_V01_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


def _build_analysis(config_path: Path) -> Tuple[Any, Any]:
    """
    Load the YAML and build a v0.1 ``HabitatAnalysis`` plus its pipeline.

    Args:
        config_path: Path to the v0.1-dialect habitat YAML.

    Returns:
        Tuple of (habitat analysis object, unfitted habitat pipeline).
    """
    from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
    from habit.core.habitat_analysis.configurator import HabitatConfigurator
    from habit.utils.random_utils import seed_numpy_global

    config = HabitatAnalysisConfig.from_file(str(config_path))
    logger = logging.getLogger("parity.v01")
    logger.setLevel(logging.WARNING)
    analysis = HabitatConfigurator(config=config, logger=logger).create_habitat_analysis()
    seed_numpy_global(config.random_state)
    pipeline = analysis._build_pipeline()
    return analysis, pipeline


def _run_individual_steps(
    pipeline: Any,
    subject_id: str,
    leg_dir: Path,
    clustering_mode: str,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run one subject through every individual-level step, snapshotting as we go.

    Args:
        pipeline: The v0.1 ``HabitatPipeline``.
        subject_id: Subject to process.
        leg_dir: Leg artefact root to write checkpoints into.
        clustering_mode: ``two_step`` / ``one_step`` / ``direct_pooling``.

    Returns:
        Tuple of (final per-subject payload, per-subject extras needed later:
        ``mask_array`` and ``voxel_unit_labels``; for ``direct_pooling`` the
        unit labels are a positional 1..N index over ROI voxels).
    """
    from habit.core.habitat_analysis.pipelines.habitat_subject_data import (
        HabitatSubjectData,
    )

    data: Any = HabitatSubjectData.empty()
    extras: Dict[str, Any] = {}
    for name, step in pipeline.individual_steps:
        data = step.transform_one(subject_id, data)
        if name == "voxel_features":
            # Checkpoint 1: raw per-voxel, per-modality feature matrix.
            pio.write_table(data.features, leg_dir / pio.CK1_DIR / f"{subject_id}.parquet")
        elif name == "individual_preprocessing":
            pio.write_table(
                data.features, leg_dir / pio.CK1B_DIR / f"{subject_id}.parquet"
            )
            if clustering_mode in ("direct_pooling", "one_step"):
                # Units entering habitat k-means = every ROI voxel. Written
                # here so CK2 matches v1's ``pipeline.units()`` (no
                # supervoxelizer) rather than the post-hoc habitat means.
                extras["mask_array"] = np.asarray(data.mask_info["mask_array"])
                n_vox = int(np.asarray(data.features).shape[0])
                extras["voxel_unit_ids"] = np.arange(1, n_vox + 1, dtype=np.int32)
                unit_table = data.features.copy()
                unit_table.insert(0, "supervoxel", extras["voxel_unit_ids"])
                pio.write_table(
                    unit_table, leg_dir / pio.CK2_DIR / f"{subject_id}.parquet"
                )
                extras["unit_table"] = unit_table
        elif name == "individual_clustering":
            extras["voxel_unit_labels"] = np.asarray(data.supervoxel_labels)
            extras["mask_array"] = np.asarray(data.mask_info["mask_array"])
            if clustering_mode == "one_step":
                # Labels are already habitat ids; record the per-subject k.
                labels = extras["voxel_unit_labels"]
                extras["chosen_k"] = int(np.max(labels)) if labels.size else 0
        elif name == "merge_supervoxel_features":
            if clustering_mode == "two_step":
                # Checkpoint 2 (per subject): the clustering-unit table.
                pio.write_table(
                    data.supervoxel_df, leg_dir / pio.CK2_DIR / f"{subject_id}.parquet"
                )
            # one_step: keep the habitat-mean table for CK6 only.
            extras["habitat_feature_table"] = data.supervoxel_df
    if clustering_mode != "direct_pooling" and data.supervoxel_df is None:
        raise RuntimeError(
            f"v0.1 individual stage produced no supervoxel_df for {subject_id}"
        )
    if "mask_array" not in extras:
        raise RuntimeError(
            f"v0.1 individual stage produced no mask extras for {subject_id}"
        )
    return data, extras


def _write_label_volumes_two_step(
    results_df: pd.DataFrame,
    extras_by_subject: Dict[str, Dict[str, Any]],
    leg_dir: Path,
) -> None:
    """
    Rebuild habitat volumes from supervoxel ids and a unit->habitat mapping.

    Args:
        results_df: Group-clustering output with ``subject``, ``supervoxel``,
            ``habitats`` columns.
        extras_by_subject: Per-subject mask and unit-label arrays.
        leg_dir: Leg artefact root.
    """
    for subject_id, extras in extras_by_subject.items():
        rows = results_df[results_df["subject"] == subject_id]
        mapping = {
            int(u): int(h)
            for u, h in zip(rows["supervoxel"].to_numpy(), rows["habitats"].to_numpy())
        }
        volume = pio.label_volume_from_units(
            extras["mask_array"], extras["voxel_unit_labels"], mapping
        )
        pio.write_array(volume, leg_dir / pio.CK5_DIR / f"{subject_id}.npy")


def _write_label_volumes_one_step(
    extras_by_subject: Dict[str, Dict[str, Any]],
    leg_dir: Path,
) -> None:
    """
    Rebuild habitat volumes when individual clustering already assigned habitats.

    Args:
        extras_by_subject: Per-subject mask and habitat-label arrays.
        leg_dir: Leg artefact root.
    """
    for subject_id, extras in extras_by_subject.items():
        labels = np.asarray(extras["voxel_unit_labels"], dtype=np.int32)
        # Identity mapping: the per-voxel labels are already habitat ids.
        mapping = {int(u): int(u) for u in np.unique(labels)}
        volume = pio.label_volume_from_units(
            extras["mask_array"], labels, mapping
        )
        pio.write_array(volume, leg_dir / pio.CK5_DIR / f"{subject_id}.npy")


def _write_label_volumes_pooling(
    results_df: pd.DataFrame,
    extras_by_subject: Dict[str, Dict[str, Any]],
    leg_dir: Path,
) -> None:
    """
    Rebuild habitat volumes from pooled-voxel habitat assignments.

    ``direct_pooling`` writes one row per ROI voxel; row order within each
    subject matches the ROI raster order used for ``voxel_unit_labels``.

    Args:
        results_df: Group-clustering output with ``subject`` and ``habitats``.
        extras_by_subject: Per-subject mask arrays.
        leg_dir: Leg artefact root.
    """
    for subject_id, extras in extras_by_subject.items():
        rows = results_df[results_df["subject"] == subject_id]
        habitats = np.asarray(rows["habitats"].to_numpy(), dtype=np.int32)
        unit_ids = np.arange(1, habitats.size + 1, dtype=np.int32)
        mapping = {int(u): int(h) for u, h in zip(unit_ids, habitats)}
        volume = pio.label_volume_from_units(
            extras["mask_array"], unit_ids, mapping
        )
        pio.write_array(volume, leg_dir / pio.CK5_DIR / f"{subject_id}.npy")


def main() -> int:
    """
    Entry point: dump checkpoints 1-6 for one config into one leg directory.

    Supports ``two_step``, ``one_step`` and ``direct_pooling``. ``one_step``
    has no cohort-level model; checkpoints 3/4 then record per-subject
    habitat tables and a stacked-centroid placeholder so the comparator still
    has artefacts to inspect.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Dump v0.1 parity checkpoints.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--leg-dir", required=True, type=Path)
    args = parser.parse_args()

    _force_v01_on_path()
    import habit

    if "habit_v01_parity" not in str(Path(habit.__file__).resolve()):
        raise RuntimeError(
            "Refusing to run: 'habit' did not resolve to the frozen v0.1 "
            f"worktree (got {habit.__file__}). Set HABIT_V01_ROOT or "
            "install the frozen tree ahead of the v1 working tree."
        )

    leg_dir = pio.ensure_leg_dir(args.leg_dir)
    started = time.perf_counter()

    analysis, pipeline = _build_analysis(args.config)
    config = analysis.config
    clustering_mode = str(config.habitat_segmentation.clustering_mode)
    subject_ids: List[str] = list(analysis.feature_service.images_paths.keys())

    roster: Dict[str, str] = {}
    payloads: Dict[str, Any] = {}
    extras_by_subject: Dict[str, Dict[str, Any]] = {}
    for subject_id in subject_ids:
        try:
            payload, extras = _run_individual_steps(
                pipeline, subject_id, leg_dir, clustering_mode
            )
        except Exception as exc:  # noqa: BLE001 - roster parity needs the reason
            roster[subject_id] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            continue
        roster[subject_id] = "success"
        payloads[subject_id] = payload
        extras_by_subject[subject_id] = extras

    if len(payloads) < 2:
        pio.write_meta(
            leg_dir,
            {
                "leg": "v0.1",
                "habit_version": habit.__version__,
                "config": str(args.config),
                "clustering_mode": clustering_mode,
                "roster": roster,
                "fatal": "fewer than two subjects survived the individual stage",
            },
        )
        return 1

    group_input: Any = payloads
    cohort_prep_frame = None
    group_clustering_step = None
    results_df = None
    combined_df = None
    for name, step in pipeline.group_steps:
        group_input = step.fit_transform(group_input)
        if name in ("combine_supervoxels", "concatenate_voxels"):
            # Checkpoint 2 (cohort): pooled units INCLUDING row order.
            combined_df = group_input
            pio.write_table(group_input, leg_dir / pio.CK2_COHORT)
        elif name == "group_preprocessing":
            cohort_prep_frame = group_input
            pio.write_table(group_input, leg_dir / pio.CK3_COHORT_PREP)
        elif name == "group_clustering":
            group_clustering_step = step
            results_df = group_input

    if combined_df is None and clustering_mode == "one_step":
        # one_step group stage is only combine_supervoxels; handled above.
        pass

    if cohort_prep_frame is None and (leg_dir / pio.CK2_COHORT).is_file():
        # No cohort preprocessing configured: the pooled table is the matrix.
        pio.write_table(
            pio.read_table(leg_dir / pio.CK2_COHORT), leg_dir / pio.CK3_COHORT_PREP
        )

    chosen_k_by_subject: Dict[str, int] = {
        sid: int(ex.get("chosen_k", -1)) for sid, ex in extras_by_subject.items()
    }

    if clustering_mode == "one_step":
        # Rebuild CK2 cohort from the per-voxel unit tables (not from
        # combine_supervoxels, which concatenates post-hoc habitat means).
        unit_rows: List[pd.DataFrame] = []
        habitat_rows: List[pd.DataFrame] = []
        for sid, extras in extras_by_subject.items():
            ut = extras["unit_table"].copy()
            if "subject" not in ut.columns:
                ut.insert(0, "subject", sid)
            else:
                ut["subject"] = sid
            unit_rows.append(ut)
            ht = extras.get("habitat_feature_table")
            if ht is not None:
                ht = ht.copy()
                if "subject" not in ht.columns:
                    ht.insert(0, "subject", sid)
                else:
                    ht["subject"] = sid
                if "habitats" not in ht.columns and "supervoxel" in ht.columns:
                    ht["habitats"] = ht["supervoxel"]
                habitat_rows.append(ht)
        # CK6 for one_step is per-voxel habitat labels (same grain as v1),
        # not the post-hoc 1-row-per-habitat mean table.
        voxel_label_rows: List[pd.DataFrame] = []
        for sid, extras in extras_by_subject.items():
            ut = extras["unit_table"].copy()
            if "subject" not in ut.columns:
                ut.insert(0, "subject", sid)
            else:
                ut["subject"] = sid
            ut["habitats"] = np.asarray(extras["voxel_unit_labels"], dtype=np.int64)
            voxel_label_rows.append(ut)
        results_df = pd.concat(voxel_label_rows, ignore_index=True)
        pio.write_table(pd.concat(unit_rows, ignore_index=True), leg_dir / pio.CK2_COHORT)
        pio.write_table(pd.concat(unit_rows, ignore_index=True), leg_dir / pio.CK3_COHORT_PREP)
        ks = [k for k in chosen_k_by_subject.values() if k > 0]
        chosen_k = int(sorted(ks)[len(ks) // 2]) if ks else 0
        feature_names = [
            c
            for c in unit_rows[0].select_dtypes(include=[np.number]).columns
            if c not in ("subject", "supervoxel", "count", "habitats")
            and not str(c).endswith("-original")
        ]
        pio.write_model(
            leg_dir,
            np.zeros((0, len(feature_names)), dtype=np.float64),
            feature_names,
        )
        _write_label_volumes_one_step(extras_by_subject, leg_dir)
    else:
        if group_clustering_step is None or results_df is None:
            raise RuntimeError("v0.1 group stage produced no clustering result")
        model = group_clustering_step.clustering_model
        centroids = np.asarray(model.cluster_centers_, dtype=np.float64)
        feature_names = [
            c
            for c in results_df.select_dtypes(include=[np.number]).columns
            if c not in ("subject", "supervoxel", "count", "habitats")
            and not str(c).endswith("-original")
        ]
        pio.write_model(leg_dir, centroids, feature_names)
        chosen_k = int(group_clustering_step.optimal_n_clusters_)
        if clustering_mode == "direct_pooling":
            _write_label_volumes_pooling(results_df, extras_by_subject, leg_dir)
        else:
            _write_label_volumes_two_step(results_df, extras_by_subject, leg_dir)

    pio.write_table(results_df, leg_dir / pio.CK6_FEATURES)

    pio.write_meta(
        leg_dir,
        {
            "leg": "v0.1",
            "habit_version": habit.__version__,
            "habit_path": str(Path(habit.__file__).resolve().parent),
            "config": str(args.config),
            "clustering_mode": clustering_mode,
            "random_state": config.random_state,
            "roster": roster,
            "subjects_ok": sorted(payloads),
            "chosen_k": chosen_k,
            "chosen_k_by_subject": chosen_k_by_subject,
            "n_centroid_features": int(
                0
                if clustering_mode == "one_step"
                else np.asarray(
                    group_clustering_step.clustering_model.cluster_centers_
                ).shape[1]
            ),
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    )
    print(f"[v0.1] done in {time.perf_counter() - started:.1f}s -> {leg_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
