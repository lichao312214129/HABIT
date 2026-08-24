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
"""Shared runners for the synthetic fast golden gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from habit.datasets import make_synthetic_cohort, make_synthetic_feature_table
from habit.spec.specs import HabitatSpec, Spec

#: Fixed habitat count for every fast golden case.
FAST_N_HABITATS = 3

#: Lightweight PyRadiomics settings used by the fast habitat-features case.
FAST_RADIOMICS_PARAMS: Dict[str, Any] = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": ["Mean", "Entropy"],
        "shape": ["VoxelVolume", "SurfaceArea"],
    },
    "setting": {"binWidth": 25, "label": 1},
}

#: Manifest / provenance JSON leaves excluded from golden comparison.
VOLATILE_JSON_LEAVES: Tuple[str, ...] = (
    "started_at",
    "finished_at",
    "created_at",
    "software",
    "runtime",
    "software_version",
)


@dataclass(frozen=True)
class FastGoldenCase:
    """
    One synthetic golden case executed through the v1 stack.

    Attributes:
        name: Baseline identifier and output subdirectory name.
        runner: Callable writing artefacts into ``out_dir`` and returning the
            in-memory study result when one exists.
        description: Human-readable summary for the baseline record.
        depends_on: Optional upstream case whose output directory must exist.
    """

    name: str
    runner: Callable[[Path], Any]
    description: str
    depends_on: Optional[str] = None


def baseline_dir() -> Path:
    """Return the committed fast baseline directory."""
    return Path(__file__).resolve().parents[1] / "baseline" / "fast"


def baseline_path(case_name: str) -> Path:
    """Return one committed fast baseline JSON file."""
    return baseline_dir() / f"{case_name}.json"


def _light_habitat_spec(*, two_step: bool) -> HabitatSpec:
    """
    Build a fast habitat specification with a fixed habitat count.

    Args:
        two_step: When ``True`` include a SLIC supervoxel stage.

    Returns:
        A fully seeded :class:`~habit.spec.specs.HabitatSpec`.
    """
    return HabitatSpec(
        name="fast_synthetic",
        voxel_feature_extractor=Spec(
            name="raw",
            params={"modalities": ["T1", "T2"], "roi": "tumor"},
        ),
        supervoxelizer=(
            Spec(name="slic", params={"n_supervoxels": 16, "compactness": 5.0})
            if two_step
            else None
        ),
        habitat_model_fitter=Spec(
            name="kmeans",
            params={"n_habitats": FAST_N_HABITATS, "n_init": 5},
        ),
        habitat_assigner=Spec(name="nearest_centroid", params={}),
        habitat_features=(Spec(name="volume"),),
        random_seed=0,
    )


def synthetic_cohort() -> Any:
    """Return the canonical fast synthetic cohort."""
    return make_synthetic_cohort(
        n_subjects=4,
        modalities=("T1", "T2"),
        shape=(32, 32, 32),
        n_subregions=3,
        rng=0,
    )


def _write_habitat_features(
    out_dir: Path,
    *,
    study_result: Any,
    cohort: Any,
) -> None:
    """
    Extract a small set of habitat feature families and write CSV artefacts.

    Args:
        out_dir: Destination directory.
        study_result: Completed two-step study.
        cohort: Cohort used for the study.
    """
    from habit.domain.habitat_features import (
        HabitatVolumeFeatures,
        IthHabitatFeatures,
        MsiHabitatFeatures,
        TraditionalRadiomicsHabitatFeatures,
    )

    maps = {habitat_map.subject_id: habitat_map for habitat_map in study_result.habitat_maps}
    subjects = {subject.subject_id: subject for subject in cohort}
    extractors = (
        ("msi_features.csv", MsiHabitatFeatures()),
        ("ith_scores.csv", IthHabitatFeatures(include_auxiliary=True)),
        ("habitat_basic_features.csv", HabitatVolumeFeatures()),
        (
            "raw_image_radiomics.csv",
            TraditionalRadiomicsHabitatFeatures(
                params=FAST_RADIOMICS_PARAMS,
                modalities=["T1"],
            ),
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, extractor in extractors:
        rows = []
        for subject_id, subject in subjects.items():
            table = extractor(subject, maps[subject_id])
            row = table.frame.iloc[0].to_dict()
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / filename, index=False)


def _slice_table(table: Any, indices: np.ndarray) -> Any:
    """
    Return a feature-table view restricted to ``indices``.

    Args:
        table: Source :class:`~habit.contracts.table.FeatureTable`.
        indices: Row indices to retain.

    Returns:
        A new table sharing column semantics with ``table``.
    """
    from habit.contracts.table import FeatureTable

    return FeatureTable(
        frame=table.frame.iloc[indices].reset_index(drop=True),
        id_columns=table.id_columns,
        feature_columns=table.feature_columns,
        outcome=table.outcome,
        provenance=table.provenance,
    )


def run_fast_ml_kfold(out_dir: Path) -> Dict[str, Any]:
    """
    Execute a minimal three-fold CV workflow on a synthetic feature table.

    Args:
        out_dir: Directory receiving ``ml_kfold_results.json`` and
            ``ml_kfold_summary.csv``.

    Returns:
        Aggregated metrics keyed by model name.
    """
    from habit.domain.classification import LogisticRegressionClassifier
    from habit.domain.evaluation import AccuracyMetric, AucMetric
    from habit.domain.outcome_access import outcome_series
    from habit.domain.pipeline import TablePipeline
    from habit.domain.split import kfold_indices

    table = make_synthetic_feature_table(
        n_rows=60,
        n_features=12,
        task="binary",
        rng=0,
    )
    labels = outcome_series(table, owner="fast_ml_kfold").to_numpy()
    metrics = [AccuracyMetric(), AucMetric()]
    classifier = LogisticRegressionClassifier(max_iter=500)
    classifier.set_random_state(0)

    fold_payloads: List[Dict[str, Any]] = []
    for fold_index, (train_idx, test_idx) in enumerate(
        kfold_indices(len(table.frame), n_splits=3, labels=labels, seed=0)
    ):
        pipeline = TablePipeline(steps=[], classifier=classifier)
        fitted = pipeline.fit(_slice_table(table, train_idx))
        test_table = _slice_table(table, test_idx)
        fold_metrics = fitted.evaluate(test_table, metrics=metrics)
        probabilities = fitted.predict_proba(test_table)
        fold_payloads.append(
            {
                "fold": fold_index,
                "metrics": fold_metrics,
                "y_true": outcome_series(test_table, owner="fast_ml_kfold").tolist(),
                "y_pred": fitted.predict(test_table).tolist(),
                "y_score": probabilities.iloc[:, 1].tolist(),
            }
        )

    aggregated_metrics = {
        metric.spec.name: float(
            np.mean([payload["metrics"][metric.spec.name] for payload in fold_payloads])
        )
        for metric in metrics
    }
    results = {
        "aggregated": {
            "LogisticRegression": {
                "metrics": aggregated_metrics,
                "auc_mean": aggregated_metrics.get("auc"),
                "auc_std": float(
                    np.std([payload["metrics"]["auc"] for payload in fold_payloads])
                ),
            }
        },
        "folds": fold_payloads,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ml_kfold_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    summary = pd.DataFrame(
        [
            {
                "model": "LogisticRegression",
                **aggregated_metrics,
            }
        ]
    )
    summary.to_csv(out_dir / "ml_kfold_summary.csv", index=False)
    return results


def _run_two_step(out_dir: Path) -> Any:
    from habit.recipes.study import Study

    cohort = synthetic_cohort()
    result = Study(spec=_light_habitat_spec(two_step=True), design="two_step").fit_predict(cohort)
    result.save(out_dir)
    return result


def _run_one_step(out_dir: Path) -> Any:
    from habit.recipes.study import Study

    cohort = synthetic_cohort()
    result = Study(spec=_light_habitat_spec(two_step=False), design="one_step").fit_predict(cohort)
    result.save(out_dir)
    return result


def _run_direct_pooling(out_dir: Path) -> Any:
    from habit.recipes.study import Study

    cohort = synthetic_cohort()
    result = Study(spec=_light_habitat_spec(two_step=False), design="direct_pooling").fit_predict(cohort)
    result.save(out_dir)
    return result


def _run_predict(out_dir: Path) -> Any:
    from habit.recipes.study import Study

    cohort = synthetic_cohort()
    spec = _light_habitat_spec(two_step=True)
    train = Study(spec=spec, design="two_step").fit_predict(cohort)
    assert train.habitat_model is not None
    result = Study.from_model(train.habitat_model, spec).predict(cohort)
    result.save(out_dir)
    return result


def _run_habitat_features(out_dir: Path) -> None:
    from habit.recipes.study import Study

    cohort = synthetic_cohort()
    study = Study(spec=_light_habitat_spec(two_step=True), design="two_step").fit_predict(cohort)
    _write_habitat_features(out_dir, study_result=study, cohort=cohort)


FAST_GOLDEN_CASES: Tuple[FastGoldenCase, ...] = (
    FastGoldenCase(
        name="habitat_two_step",
        runner=_run_two_step,
        description="two_step on a synthetic 4-subject cohort with 3 habitats",
    ),
    FastGoldenCase(
        name="habitat_one_step",
        runner=_run_one_step,
        description="one_step on the synthetic cohort with 3 habitats per subject",
    ),
    FastGoldenCase(
        name="habitat_direct_pooling",
        runner=_run_direct_pooling,
        description="direct_pooling on the synthetic cohort with 3 habitats",
    ),
    FastGoldenCase(
        name="habitat_two_step_predict",
        runner=_run_predict,
        description="apply a fitted two-step model back onto the training cohort",
    ),
    FastGoldenCase(
        name="habitat_features",
        runner=_run_habitat_features,
        description="light habitat feature families on synthetic two-step maps",
    ),
    FastGoldenCase(
        name="ml_kfold",
        runner=run_fast_ml_kfold,
        description="three-fold CV on a synthetic binary feature table",
    ),
)


def _is_volatile_json_leaf(leaf_path: str) -> bool:
    """
    Return whether a flattened JSON path should be ignored in comparisons.

    Args:
        leaf_path: Dotted path inside a fingerprinted JSON document.

    Returns:
        ``True`` when the leaf records environment-specific metadata.
    """
    terminal = leaf_path.rsplit(".", 1)[-1]
    if terminal in VOLATILE_JSON_LEAVES:
        return True
    return ".software." in leaf_path or leaf_path.endswith(".software")


def scrub_fingerprint_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove volatile JSON leaves from one artefact fingerprint.

    Args:
        record: Fingerprint produced by ``fingerprint_output_dir``.

    Returns:
        A copy safe to compare across runs and machines.
    """
    if record.get("kind") != "json":
        return record
    numeric = {
        key: value
        for key, value in (record.get("numeric") or {}).items()
        if not _is_volatile_json_leaf(key)
    }
    literal = {
        key: value
        for key, value in (record.get("literal") or {}).items()
        if not _is_volatile_json_leaf(key)
    }
    return {**record, "numeric": numeric, "literal": literal}


def scrub_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scrub every fingerprint inside a captured golden record.

    Args:
        record: Full case record with ``fingerprints``.

    Returns:
        A deep-scrubbed copy suitable for ``compare_records``.
    """
    fingerprints = {
        key: scrub_fingerprint_record(value)
        for key, value in record.get("fingerprints", {}).items()
    }
    return {**record, "fingerprints": fingerprints}


def run_case(case: FastGoldenCase, out_dir: Path) -> Dict[str, Any]:
    """
    Execute one fast golden case and fingerprint its output directory.

    Args:
        case: Case definition.
        out_dir: Fresh output directory for the run.

    Returns:
        Baseline-shaped record for the case.
    """
    from scripts.make_golden_baseline import fingerprint_output_dir

    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case.runner(out_dir)
    record = fingerprint_output_dir(out_dir)
    record["case"] = case.name
    record["description"] = case.description
    record["synthetic"] = True
    return record


def compare_fast_records(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
) -> List[str]:
    """
    Compare two fast golden records after scrubbing volatile JSON leaves.

    Args:
        baseline: Committed baseline document.
        current: Freshly captured record.

    Returns:
        Human-readable differences; empty when the run matches.
    """
    from scripts.make_golden_baseline import compare_records

    return compare_records(scrub_record(baseline), scrub_record(current))
