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
"""Dump layered parity checkpoints from the CURRENT v1.0 working tree.

Mirrors ``dump_v01.py`` stage for stage, but through v1's own public API:
the same v0.1-dialect YAML is translated by ``LegacyConfigAdapter`` into a
``HabitatSpec``, the spec is assembled by ``build_habitat_components``, and
the cohort-level fit reproduces ``habit.recipes.habitat._fit_cohort_model``
exactly (pool -> fit chain -> transform -> fit model).

Run from the repository root with the v1 tree on ``sys.path``::

    python developer/parity/dump_v1.py --config <yaml> --leg-dir <dir>
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
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
# The py310 env also carries a non-editable ``habit`` wheel in site-packages.
# Putting the working tree first is what makes this leg test the CURRENT code
# rather than whatever build happens to be installed.
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT))

import _parity_io as pio  # noqa: E402


def _load_spec_and_cohort(config_path: Path) -> Tuple[Any, Any, Any]:
    """
    Translate the v0.1-dialect YAML into a v1 spec and load its cohort.

    Args:
        config_path: Path to the shared habitat YAML.

    Returns:
        Tuple of (config object, ``HabitatSpec``, ``Cohort``).
    """
    from habit.api.habitat import HabitatAnalysisConfig
    from habit.recipes.yaml_runner import _load_habitat_cohort
    from habit.spec.legacy import LegacyConfigAdapter
    from habit.spec.specs import HabitatSpec

    config = HabitatAnalysisConfig.from_file(str(config_path))
    document = LegacyConfigAdapter().translate(config.model_dump(), "habitat").document
    spec_payload = document.get("spec")
    if spec_payload is None:
        raise RuntimeError("The YAML translated to no habitat spec.")
    spec = HabitatSpec.from_dict(spec_payload)
    cohort = _load_habitat_cohort(config, spec, logger=None)
    return config, spec, cohort


def _dump_subject_features(
    fit_pipeline: Any,
    subject: Any,
    leg_dir: Path,
) -> Any:
    """
    Write checkpoints 1 / 1b / 2 for one subject and return its units.

    Stage-1 runs once: the dump mirrors ``SubjectPipeline.units`` so the
    voxel extractor is not invoked a second time just to materialise the
    checkpoint tables (that double call dwarfed wall time on radiomics and
    still mattered for raw one_step / direct_pooling dumps).

    Args:
        fit_pipeline: Subject pipeline without an assigner.
        subject: Subject to process.
        leg_dir: Leg artefact root.

    Returns:
        The subject's clustering units (``Supervoxelization``).
    """
    from habit.domain.pipeline import voxel_units

    subject_id = subject.subject_id
    field = fit_pipeline.voxel_feature_extractor(subject)
    pio.write_table(
        field.feature_frame(), leg_dir / pio.CK1_DIR / f"{subject_id}.parquet"
    )
    original_field = field
    if fit_pipeline.voxel_feature_preprocessor is not None:
        chain = fit_pipeline.voxel_feature_preprocessor
        field = field.with_feature_frame(
            chain(field.feature_frame()),
            produced_by="feature_preprocessing.subject.voxel",
            spec_fingerprint=chain.spec.fingerprint(),
        )
    pio.write_table(
        field.feature_frame(), leg_dir / pio.CK1B_DIR / f"{subject_id}.parquet"
    )
    if fit_pipeline.supervoxelizer is None:
        units = voxel_units(field)
    else:
        # Keep the full pipeline path for two_step so statistical
        # supervoxel extractors still see ``original`` vs ``working``.
        units = fit_pipeline.supervoxelizer(field)
        if fit_pipeline.supervoxel_feature_extractor is not None:
            binder = getattr(
                fit_pipeline.supervoxel_feature_extractor, "bind_fields", None
            )
            if callable(binder):
                binder(working=field, original=original_field)
            units = fit_pipeline.supervoxel_feature_extractor(subject, units)
        if fit_pipeline.supervoxel_feature_preprocessor is not None:
            chain = fit_pipeline.supervoxel_feature_preprocessor
            units = units.with_feature_frame(
                chain(units.feature_frame()),
                produced_by="feature_preprocessing.subject.supervoxel",
                spec_fingerprint=chain.spec.fingerprint(),
            )
    pio.write_table(
        units.feature_frame(), leg_dir / pio.CK2_DIR / f"{subject_id}.parquet"
    )
    return units


def _run_cohort_design(
    *,
    habit: Any,
    config: Any,
    spec: Any,
    cohort: Any,
    components: Any,
    ok_subjects: List[Any],
    units_list: List[Any],
    roster: Dict[str, str],
    leg_dir: Path,
    started: float,
    config_path: Path,
) -> int:
    """
    Dump cohort-level checkpoints for ``two_step`` / ``direct_pooling``.

    Args:
        habit: Imported ``habit`` module (for version metadata).
        config: Loaded habitat analysis config.
        spec: Translated ``HabitatSpec``.
        cohort: Full cohort (used for the fit cohort name).
        components: Assembled habitat components.
        ok_subjects: Subjects that survived the individual stage.
        units_list: Matching clustering units, in the same order.
        roster: Per-subject success / failure strings.
        leg_dir: Leg artefact root.
        started: ``perf_counter`` value at run start.
        config_path: Path to the YAML that drove this run.

    Returns:
        Process exit code.
    """
    from habit.contracts.subject import Cohort
    from habit.domain.protocols import Seedable

    pooled = pd.concat(
        [units.feature_frame() for units in units_list], ignore_index=True
    )
    pooled_ids = pd.DataFrame(
        {
            "subject": np.repeat(
                [s.subject_id for s in ok_subjects],
                [len(u.feature_frame()) for u in units_list],
            ),
            "supervoxel": np.concatenate(
                [np.asarray(u.features.index) for u in units_list]
            ),
        }
    )
    pio.write_table(pd.concat([pooled_ids, pooled], axis=1), leg_dir / pio.CK2_COHORT)

    chain = components.cohort_chain
    working = list(units_list)
    if chain is not None:
        chain.fit(pooled)
        working = [
            units.with_feature_frame(
                chain.transform(units.feature_frame()),
                produced_by="feature_preprocessing.cohort",
                spec_fingerprint=chain.spec.fingerprint(),
            )
            for units in units_list
        ]
    cohort_prep = pd.concat(
        [units.feature_frame() for units in working], ignore_index=True
    )
    pio.write_table(
        pd.concat([pooled_ids, cohort_prep], axis=1), leg_dir / pio.CK3_COHORT_PREP
    )

    fit_cohort = Cohort(subjects=tuple(ok_subjects), name=cohort.name)
    model = components.fitter.fit(working, cohort=fit_cohort)
    if chain is not None:
        model = model.with_cohort_preprocessing(chain.state, chain.spec.to_dict())
    pio.write_model(
        leg_dir, np.asarray(model.centroids, dtype=np.float64), model.feature_names
    )

    assigner = model.assigner(
        spec.habitat_assigner.name, **spec.habitat_assigner.params
    )
    if isinstance(assigner, Seedable) and spec.random_seed is not None:
        assigner.set_random_state(spec.random_seed)
    predict_pipeline = components.pipeline(assigner=assigner)

    label_rows: List[pd.DataFrame] = []
    for subject, units in zip(ok_subjects, units_list):
        habitat_map, prepared = predict_pipeline.assign(units)
        pio.write_array(
            np.asarray(habitat_map.label_array, dtype=np.int32),
            leg_dir / pio.CK5_DIR / f"{subject.subject_id}.npy",
        )
        frame = prepared.feature_frame().copy()
        frame.insert(0, "supervoxel", np.asarray(prepared.features.index))
        frame.insert(0, "subject", subject.subject_id)
        frame["habitats"] = _unit_habitat_labels(prepared, habitat_map)
        label_rows.append(frame)
    pio.write_table(pd.concat(label_rows, ignore_index=True), leg_dir / pio.CK6_FEATURES)

    pio.write_meta(
        leg_dir,
        {
            "leg": "v1.0",
            "habit_version": habit.__version__,
            "habit_path": str(Path(habit.__file__).resolve().parent),
            "config": str(config_path),
            "clustering_mode": config.habitat_segmentation.clustering_mode,
            "random_state": config.random_state,
            "roster": roster,
            "subjects_ok": sorted(s.subject_id for s in ok_subjects),
            "chosen_k": int(model.n_habitats),
            "n_centroid_features": int(np.asarray(model.centroids).shape[1]),
            "model_id": model.model_id,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    )
    return 0


def _run_one_step_design(
    *,
    habit: Any,
    config: Any,
    spec: Any,
    components: Any,
    ok_subjects: List[Any],
    units_list: List[Any],
    roster: Dict[str, str],
    leg_dir: Path,
    started: float,
    config_path: Path,
) -> int:
    """
    Dump per-subject habitat definitions for ``one_step``.

    Mirrors ``habit.recipes.habitat._DefineAndLabelWithinSubject``: each
    subject's units are fitted independently, so there is no cohort model.

    Args:
        habit: Imported ``habit`` module.
        config: Loaded habitat analysis config.
        spec: Translated ``HabitatSpec``.
        components: Assembled habitat components.
        ok_subjects: Subjects that survived feature extraction.
        units_list: Matching clustering units.
        roster: Per-subject success / failure strings.
        leg_dir: Leg artefact root.
        started: ``perf_counter`` value at run start.
        config_path: Path to the YAML that drove this run.

    Returns:
        Process exit code.
    """
    from habit.domain.protocols import Seedable

    pooled_ids_frames: List[pd.DataFrame] = []
    pooled_feat_frames: List[pd.DataFrame] = []
    label_rows: List[pd.DataFrame] = []
    chosen_k_by_subject: Dict[str, int] = {}

    for subject, units in zip(ok_subjects, units_list):
        model = components.fitter.fit([units])
        chosen_k_by_subject[subject.subject_id] = int(model.n_habitats)
        assigner = model.assigner(
            spec.habitat_assigner.name, **spec.habitat_assigner.params
        )
        if isinstance(assigner, Seedable) and spec.random_seed is not None:
            assigner.set_random_state(spec.random_seed)
        pipeline = components.pipeline(assigner=assigner)
        habitat_map, prepared = pipeline.assign(units)
        pio.write_array(
            np.asarray(habitat_map.label_array, dtype=np.int32),
            leg_dir / pio.CK5_DIR / f"{subject.subject_id}.npy",
        )
        frame = prepared.feature_frame().copy()
        frame.insert(0, "supervoxel", np.asarray(prepared.features.index))
        frame.insert(0, "subject", subject.subject_id)
        frame["habitats"] = _unit_habitat_labels(prepared, habitat_map)
        label_rows.append(frame)
        feat = units.feature_frame()
        pooled_feat_frames.append(feat)
        pooled_ids_frames.append(
            pd.DataFrame(
                {
                    "subject": [subject.subject_id] * len(feat),
                    "supervoxel": np.asarray(units.features.index),
                }
            )
        )

    pooled_ids = pd.concat(pooled_ids_frames, ignore_index=True)
    pooled = pd.concat(pooled_feat_frames, ignore_index=True)
    pio.write_table(pd.concat([pooled_ids, pooled], axis=1), leg_dir / pio.CK2_COHORT)
    # one_step applies no cohort preprocessing.
    pio.write_table(pd.concat([pooled_ids, pooled], axis=1), leg_dir / pio.CK3_COHORT_PREP)

    ks = [k for k in chosen_k_by_subject.values() if k > 0]
    chosen_k = int(sorted(ks)[len(ks) // 2]) if ks else 0
    feature_names = list(pooled.columns)
    pio.write_model(
        leg_dir,
        np.zeros((0, len(feature_names)), dtype=np.float64),
        feature_names,
    )
    pio.write_table(pd.concat(label_rows, ignore_index=True), leg_dir / pio.CK6_FEATURES)

    pio.write_meta(
        leg_dir,
        {
            "leg": "v1.0",
            "habit_version": habit.__version__,
            "habit_path": str(Path(habit.__file__).resolve().parent),
            "config": str(config_path),
            "clustering_mode": "one_step",
            "random_state": config.random_state,
            "roster": roster,
            "subjects_ok": sorted(s.subject_id for s in ok_subjects),
            "chosen_k": chosen_k,
            "chosen_k_by_subject": chosen_k_by_subject,
            "n_centroid_features": 0,
            "model_id": None,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    )
    return 0


def main() -> int:
    """
    Entry point: dump checkpoints 1-6 for one config into one leg directory.

    ``two_step`` / ``direct_pooling`` follow the cohort-fit recipe path;
    ``one_step`` follows the per-subject ``_DefineAndLabelWithinSubject`` path.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="Dump v1.0 parity checkpoints.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--leg-dir", required=True, type=Path)
    args = parser.parse_args()

    import habit

    resolved = Path(habit.__file__).resolve()
    if resolved.parent != _REPO_ROOT / "habit":
        raise RuntimeError(
            "Refusing to run: 'habit' did not resolve to the working tree "
            f"(got {resolved}). A stale wheel in site-packages would silently "
            "make this leg test the wrong code."
        )

    from habit.domain.assembly import build_habitat_components

    leg_dir = pio.ensure_leg_dir(args.leg_dir)
    started = time.perf_counter()

    config, spec, cohort = _load_spec_and_cohort(args.config)
    clustering_mode = str(config.habitat_segmentation.clustering_mode)
    components = build_habitat_components(spec)
    fit_pipeline = components.pipeline(assigner=None)

    roster: Dict[str, str] = {}
    ok_subjects: List[Any] = []
    units_list: List[Any] = []
    for subject in cohort:
        subject_id = subject.subject_id
        try:
            units = _dump_subject_features(fit_pipeline, subject, leg_dir)
        except Exception as exc:  # noqa: BLE001 - roster parity needs the reason
            roster[subject_id] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            continue
        roster[subject_id] = "success"
        ok_subjects.append(subject)
        units_list.append(units)

    if len(units_list) < 2:
        pio.write_meta(
            leg_dir,
            {
                "leg": "v1.0",
                "habit_version": habit.__version__,
                "config": str(args.config),
                "clustering_mode": clustering_mode,
                "roster": roster,
                "fatal": "fewer than two subjects survived the individual stage",
            },
        )
        return 1

    if clustering_mode == "one_step":
        code = _run_one_step_design(
            habit=habit,
            config=config,
            spec=spec,
            components=components,
            ok_subjects=ok_subjects,
            units_list=units_list,
            roster=roster,
            leg_dir=leg_dir,
            started=started,
            config_path=args.config,
        )
    else:
        code = _run_cohort_design(
            habit=habit,
            config=config,
            spec=spec,
            cohort=cohort,
            components=components,
            ok_subjects=ok_subjects,
            units_list=units_list,
            roster=roster,
            leg_dir=leg_dir,
            started=started,
            config_path=args.config,
        )

    print(f"[v1.0] done in {time.perf_counter() - started:.1f}s -> {leg_dir}")
    return code


def _unit_habitat_labels(units: Any, habitat_map: Any) -> np.ndarray:
    """
    Read back the habitat label assigned to each clustering unit.

    The v0.1 leg records a per-unit ``habitats`` column, so the v1 leg needs
    the same table shape. The label of a unit is the habitat value of any of
    its voxels, taken from the label volume.

    Implementation note: one_step / direct_pooling use ``voxel_units``, so
    every ROI voxel is its own unit id inside a full-volume ``label_array``
    (often ~10^6–10^7 voxels with thousands of unique ids). A Python loop
    ``for unit_id: label_array == unit_id`` is O(n_units * volume) and was
    the entire ~20s→90s+ wall-time gap in the dump harness — not a recipe
    regression. Build a dense unit→habitat lookup in one O(volume) pass.

    Args:
        units: Clustering units after cohort preprocessing.
        habitat_map: The habitat label image produced for the same subject.

    Returns:
        One habitat label per unit, in ``units.feature_frame()`` row order.
    """
    unit_ids = np.asarray(units.features.index, dtype=np.int64)
    unit_labels = np.asarray(units.label_array).ravel()
    habitat_labels = np.asarray(habitat_map.label_array).ravel()
    if unit_ids.size == 0:
        return np.asarray([], dtype=np.int64)
    lookup_size = int(max(int(unit_ids.max()), int(unit_labels.max()), 0)) + 1
    lookup = np.zeros(lookup_size, dtype=np.int64)
    # First-hit wins; all voxels of a unit share one habitat after assign.
    nz = unit_labels != 0
    lookup[unit_labels[nz]] = habitat_labels[nz].astype(np.int64, copy=False)
    return lookup[unit_ids]


if __name__ == "__main__":
    raise SystemExit(main())
