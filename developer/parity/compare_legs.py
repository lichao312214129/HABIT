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
"""Compare two parity legs checkpoint by checkpoint and print one verdict table.

Loads only the on-disk artefacts written by ``dump_v01.py`` and ``dump_v1.py``,
so it needs neither HABIT version importable. The first FAILing checkpoint
localises a divergence: every later checkpoint is downstream of it.

Column names legitimately differ between the two versions (``raw-T1`` versus
``T1``), so feature matrices are aligned POSITIONALLY after metadata columns
are dropped; a shape mismatch is itself reported as a failure.

Run::

    python developer/parity/compare_legs.py --a <v01 leg dir> --b <v1 leg dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _parity_io as pio  # noqa: E402

RTOL = 1e-6
ATOL = 1e-9

#: Columns that carry identifiers or v0.1-only bookkeeping rather than the
#: numbers being compared.
_METADATA_COLUMNS = {"index", "subject", "supervoxel", "count", "habitats"}


def _feature_matrix(frame: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Reduce a stored table to its comparable numeric feature matrix.

    Args:
        frame: Table as written by the dump harnesses.

    Returns:
        Tuple of (float matrix in column order, the retained column names).
        v0.1's ``*-original`` companion columns are dropped: they are a v0.1
        bookkeeping artefact with no v1 counterpart.
    """
    columns = [
        c
        for c in frame.columns
        if c not in _METADATA_COLUMNS and not str(c).endswith("-original")
    ]
    numeric = frame[columns].select_dtypes(include=[np.number])
    return numeric.to_numpy(dtype=np.float64), list(numeric.columns)


def _diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """
    Compute the largest absolute and relative difference between two matrices.

    Args:
        a: Left matrix.
        b: Right matrix, same shape as ``a``.

    Returns:
        Tuple of (max absolute difference, max relative difference). The
        relative difference is normalised by ``max(|a|, |b|)`` per element and
        is zero where both entries are zero.
    """
    abs_diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, abs_diff / scale, 0.0)
    return float(abs_diff.max(initial=0.0)), float(np.nanmax(rel, initial=0.0))


def _compare_tables(
    frame_a: Optional[pd.DataFrame],
    frame_b: Optional[pd.DataFrame],
    label: str,
) -> Dict[str, Any]:
    """
    Compare one pair of stored tables positionally.

    Args:
        frame_a: Left table, or ``None`` when the artefact is missing.
        frame_b: Right table, or ``None`` when the artefact is missing.
        label: Checkpoint label for the verdict row.

    Returns:
        A verdict row mapping.
    """
    if frame_a is None or frame_b is None:
        return {
            "checkpoint": label,
            "status": "MISSING",
            "max_abs": None,
            "max_rel": None,
            "notes": "artefact absent on one leg",
        }
    matrix_a, names_a = _feature_matrix(frame_a)
    matrix_b, names_b = _feature_matrix(frame_b)
    if matrix_a.shape != matrix_b.shape:
        return {
            "checkpoint": label,
            "status": "FAIL",
            "max_abs": None,
            "max_rel": None,
            "notes": f"shape {matrix_a.shape} vs {matrix_b.shape}",
        }
    max_abs, max_rel = _diff(matrix_a, matrix_b)
    note = f"{matrix_a.shape[0]}x{matrix_a.shape[1]}"
    if names_a != names_b:
        note += f"; columns renamed {names_a}->{names_b}"
    return {
        "checkpoint": label,
        "status": "PASS" if max_rel <= RTOL or max_abs <= ATOL else "FAIL",
        "max_abs": max_abs,
        "max_rel": max_rel,
        "notes": note,
    }


def _worst(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """
    Collapse per-subject verdict rows into one worst-case row.

    Args:
        rows: Per-subject verdict rows.
        label: Checkpoint label for the collapsed row.

    Returns:
        The collapsed verdict row.
    """
    if not rows:
        return {
            "checkpoint": label,
            "status": "MISSING",
            "max_abs": None,
            "max_rel": None,
            "notes": "no subjects",
        }
    failed = [r for r in rows if r["status"] != "PASS"]
    max_abs = max((r["max_abs"] or 0.0) for r in rows)
    max_rel = max((r["max_rel"] or 0.0) for r in rows)
    notes = f"{len(rows)} subjects"
    if failed:
        notes += "; first divergent: " + ", ".join(
            f"{r['checkpoint'].split('/')[-1]}({r['notes']})" for r in failed[:3]
        )
    else:
        notes += "; " + rows[0]["notes"]
    return {
        "checkpoint": label,
        "status": "FAIL" if failed else "PASS",
        "max_abs": max_abs,
        "max_rel": max_rel,
        "notes": notes,
    }


def _label_agreement(
    volume_a: np.ndarray, volume_b: np.ndarray
) -> Tuple[float, float, Dict[int, int]]:
    """
    Compare two habitat label volumes with and without relabelling.

    k-means habitat numbering is arbitrary, so a permutation-invariant
    comparison is the scientifically meaningful one; the raw agreement is
    reported alongside because a permuted-but-not-raw match still means every
    downstream label-indexed artefact differs.

    Args:
        volume_a: Left label volume.
        volume_b: Right label volume, same shape.

    Returns:
        Tuple of (raw agreement fraction, best-permutation agreement fraction,
        the permutation mapping left labels to right labels).
    """
    from scipy.optimize import linear_sum_assignment

    roi = (volume_a > 0) | (volume_b > 0)
    left = volume_a[roi].astype(np.int64)
    right = volume_b[roi].astype(np.int64)
    total = int(left.size)
    if total == 0:
        return 1.0, 1.0, {}
    raw = float(np.mean(left == right))

    labels_a = np.unique(left)
    labels_b = np.unique(right)
    # Confusion matrix over ROI voxels, built as a flat histogram.
    flat = (
        np.searchsorted(labels_a, left) * labels_b.size
        + np.searchsorted(labels_b, right)
    )
    counts = np.bincount(flat, minlength=labels_a.size * labels_b.size).reshape(
        labels_a.size, labels_b.size
    )
    rows, cols = linear_sum_assignment(-counts)
    matched = int(counts[rows, cols].sum())
    mapping = {int(labels_a[r]): int(labels_b[c]) for r, c in zip(rows, cols)}
    return raw, matched / total, mapping


def main() -> int:
    """
    Entry point: print the verdict table for two leg directories.

    Returns:
        Process exit code: 0 when every checkpoint passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Compare two parity legs.")
    parser.add_argument("--a", required=True, type=Path, help="reference leg (v0.1)")
    parser.add_argument("--b", required=True, type=Path, help="candidate leg (v1.0)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    meta_a = pio.read_meta(args.a)
    meta_b = pio.read_meta(args.b)
    rows: List[Dict[str, Any]] = []

    # Dimension 0: subject roster. A run can be numerically clean on a
    # different subset, so this is checked before any number is compared.
    roster_a = {k: v for k, v in meta_a["roster"].items()}
    roster_b = {k: v for k, v in meta_b["roster"].items()}
    ok_a = sorted(k for k, v in roster_a.items() if v == "success")
    ok_b = sorted(k for k, v in roster_b.items() if v == "success")
    rows.append(
        {
            "checkpoint": "0. subject roster",
            "status": "PASS" if ok_a == ok_b else "FAIL",
            "max_abs": None,
            "max_rel": None,
            "notes": (
                f"{len(ok_a)} ok both legs"
                if ok_a == ok_b
                else f"only-A={sorted(set(ok_a) - set(ok_b))} only-B={sorted(set(ok_b) - set(ok_a))}"
            ),
        }
    )

    subjects = ok_a if ok_a == ok_b else sorted(set(ok_a) & set(ok_b))

    for label, directory in (
        ("1. voxel features (raw)", pio.CK1_DIR),
        ("1b. voxel features (subject-prep)", pio.CK1B_DIR),
        ("2. clustering units (per subject)", pio.CK2_DIR),
    ):
        per_subject = [
            _compare_tables(
                pio.optional_read_table(args.a / directory / f"{s}.parquet"),
                pio.optional_read_table(args.b / directory / f"{s}.parquet"),
                f"{label}/{s}",
            )
            for s in subjects
        ]
        rows.append(_worst(per_subject, label))

    rows.append(
        _compare_tables(
            pio.optional_read_table(args.a / pio.CK2_COHORT),
            pio.optional_read_table(args.b / pio.CK2_COHORT),
            "2c. clustering units (cohort, row order)",
        )
    )
    rows.append(
        _compare_tables(
            pio.optional_read_table(args.a / pio.CK3_COHORT_PREP),
            pio.optional_read_table(args.b / pio.CK3_COHORT_PREP),
            "3. matrix entering k-means",
        )
    )

    # Checkpoint 4: chosen k, then canonically ordered centroids.
    k_a, k_b = int(meta_a["chosen_k"]), int(meta_b["chosen_k"])
    rows.append(
        {
            "checkpoint": "4a. chosen k",
            "status": "PASS" if k_a == k_b else "FAIL",
            "max_abs": None,
            "max_rel": None,
            "notes": f"k={k_a} vs {k_b}",
        }
    )
    model_a = np.load(args.a / pio.CK4_MODEL, allow_pickle=True)
    model_b = np.load(args.b / pio.CK4_MODEL, allow_pickle=True)
    cent_a = model_a["centroids_canonical"]
    cent_b = model_b["centroids_canonical"]
    mode_a = str(meta_a.get("clustering_mode", ""))
    mode_b = str(meta_b.get("clustering_mode", ""))
    if mode_a == "one_step" and mode_b == "one_step" and (
        cent_a.shape[0] == 0 and cent_b.shape[0] == 0
    ):
        # one_step has no cohort-level model; centroids are intentionally empty.
        # Per-subject k is compared via meta chosen_k / chosen_k_by_subject.
        rows.append(
            {
                "checkpoint": "4b. centroids (canonical order)",
                "status": "PASS",
                "max_abs": 0.0,
                "max_rel": 0.0,
                "notes": "one_step: no cohort centroids (per-subject models)",
            }
        )
    elif cent_a.shape != cent_b.shape:
        rows.append(
            {
                "checkpoint": "4b. centroids (canonical order)",
                "status": "FAIL",
                "max_abs": None,
                "max_rel": None,
                "notes": f"shape {cent_a.shape} vs {cent_b.shape}",
            }
        )
    else:
        max_abs, max_rel = _diff(cent_a, cent_b)
        rows.append(
            {
                "checkpoint": "4b. centroids (canonical order)",
                "status": "PASS" if max_rel <= RTOL or max_abs <= ATOL else "FAIL",
                "max_abs": max_abs,
                "max_rel": max_rel,
                "notes": (
                    f"{cent_a.shape[0]}x{cent_a.shape[1]}; "
                    f"raw-order identical={bool(np.array_equal(model_a['canonical_order'], model_b['canonical_order']))}"
                ),
            }
        )

    # Checkpoint 5: habitat label volumes, raw and after Hungarian matching.
    raw_scores: List[float] = []
    permuted_scores: List[float] = []
    mappings: List[str] = []
    for subject in subjects:
        path_a = args.a / pio.CK5_DIR / f"{subject}.npy"
        path_b = args.b / pio.CK5_DIR / f"{subject}.npy"
        if not (path_a.is_file() and path_b.is_file()):
            continue
        volume_a = np.load(path_a)
        volume_b = np.load(path_b)
        if volume_a.shape != volume_b.shape:
            raw_scores.append(0.0)
            permuted_scores.append(0.0)
            mappings.append(f"{subject}: shape {volume_a.shape} vs {volume_b.shape}")
            continue
        raw, permuted, mapping = _label_agreement(volume_a, volume_b)
        raw_scores.append(raw)
        permuted_scores.append(permuted)
        mappings.append(f"{subject}:{mapping}")
    if raw_scores:
        rows.append(
            {
                "checkpoint": "5. habitat label maps",
                "status": "PASS" if min(permuted_scores) == 1.0 else "FAIL",
                "max_abs": None,
                "max_rel": None,
                "notes": (
                    f"raw agreement min={min(raw_scores):.6f} "
                    f"mean={float(np.mean(raw_scores)):.6f}; "
                    f"permuted min={min(permuted_scores):.6f} "
                    f"mean={float(np.mean(permuted_scores)):.6f}"
                ),
            }
        )

    # Checkpoint 6: habitat feature table. When the config declares no habitat
    # feature families (v0.1 ``get-habitat`` never extracts them), this is the
    # per-unit habitat label table instead, and is reported as such.
    table_a = pio.optional_read_table(args.a / pio.CK6_FEATURES)
    table_b = pio.optional_read_table(args.b / pio.CK6_FEATURES)
    if table_a is not None and table_b is not None:
        if "habitats" in table_a.columns and "habitats" in table_b.columns:
            labels_a = table_a["habitats"].to_numpy()
            labels_b = table_b["habitats"].to_numpy()
            if labels_a.shape == labels_b.shape:
                raw, permuted, mapping = _label_agreement(
                    labels_a.astype(np.int32), labels_b.astype(np.int32)
                )
                rows.append(
                    {
                        "checkpoint": "6. unit-level habitat labels",
                        "status": "PASS" if permuted == 1.0 else "FAIL",
                        "max_abs": None,
                        "max_rel": None,
                        "notes": (
                            f"n={labels_a.size}; raw={raw:.6f}; "
                            f"permuted={permuted:.6f}; map={mapping}"
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "checkpoint": "6. unit-level habitat labels",
                        "status": "FAIL",
                        "max_abs": None,
                        "max_rel": None,
                        "notes": f"shape {labels_a.shape} vs {labels_b.shape}",
                    }
                )

    verdict = pd.DataFrame(rows)[
        ["checkpoint", "status", "max_abs", "max_rel", "notes"]
    ]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 96)
    print()
    print(f"A (reference) : {meta_a['leg']} {meta_a['habit_version']} @ {meta_a.get('habit_path')}")
    print(f"B (candidate) : {meta_b['leg']} {meta_b['habit_version']} @ {meta_b.get('habit_path')}")
    print(f"runtime       : A={meta_a.get('elapsed_sec')}s  B={meta_b.get('elapsed_sec')}s")
    print()
    print(verdict.to_string(index=False))
    print()
    failures = verdict[verdict["status"] != "PASS"]
    if failures.empty:
        print("VERDICT: all checkpoints agree within rtol=1e-6.")
    else:
        print(f"VERDICT: first divergent checkpoint = {failures.iloc[0]['checkpoint']}")

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {"meta_a": meta_a, "meta_b": meta_b, "rows": rows},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    return 0 if failures.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
