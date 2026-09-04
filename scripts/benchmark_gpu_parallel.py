#!/usr/bin/env python3
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
"""Benchmark serial vs process-pool habitat runs across CPU / GPU layouts.

Scenarios covered (pass ``--scenario`` or run ``--all``):

* ``cpu0`` — 0 GPU + multi-CPU (``CUDA_VISIBLE_DEVICES=-1``)
* ``gpu1`` — 1 GPU visible; workers may share or fall back to CPU
* ``gpuN`` — N GPUs with one worker per GPU (``cap_workers_to_gpu_pool``)

Example (cloud, 5x RTX)::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4 python scripts/benchmark_gpu_parallel.py --all --n-subjects 16 --shape 80,80,48 --workers 1,2,4,5,8

Prints a Markdown table (wall time, subjects/min, peak RSS).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from habit.spec import HabitatSpec

try:
    import resource  # Unix peak RSS; optional on Windows
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

# Limit child process BLAS/OpenMP threads to 1 to prevent thread oversubscription
# and memory contention on high-core hosts (e.g. 144 logical cores).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Ensure repo root is importable when run as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass(frozen=True)
class BenchRow:
    """One measured configuration."""

    scenario: str
    device_note: str
    workers: int
    wall_s: float
    subjects_per_min: float
    peak_rss_mb: float
    n_maps: int
    ok: bool
    error: str = ""


def _parse_int_list(raw: str) -> List[int]:
    """Parse ``1,2,4`` into integers."""
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_shape(raw: str) -> Tuple[int, int, int]:
    """Parse ``Z,Y,X`` shape."""
    parts = [int(p.strip()) for p in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be Z,Y,X")
    return parts[0], parts[1], parts[2]


def _detect_gpus() -> int:
    """Return ``torch.cuda.device_count()`` or 0."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return 0
    return 0


def _peak_rss_mb() -> float:
    """Best-effort peak RSS in MiB (Linux ``ru_maxrss`` is KiB)."""
    if resource is None:
        return float("nan")
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux: KiB; macOS: bytes. Prefer Linux cloud hosts.
        rss = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:  # noqa: BLE001
        return float("nan")


def _build_spec(
    *,
    workload: str = "voxel_radiomics",
    use_gpu: bool = True,
    seed: int = 21,
):
    """Build HabitatSpec declaring what to compute.

    Args:
        workload: 'voxel_radiomics' (dense 3D texture features via CUDA matrices
            and TorchRadiomics) or 'raw' (CPU-bound baseline).
        use_gpu: When True and workload is 'voxel_radiomics', enables CUDA
            matrix construction and TorchRadiomics GPU kernels.
        seed: Random seed for partition and k-means clustering.
    """
    from habit.spec import HabitatSpec, Spec, Stage

    if workload == "voxel_radiomics":
        extract_stage = Stage(
            "extract_voxel_features",
            Spec(
                "voxel_radiomics",
                {
                    "modalities": ["T1"],
                    "roi": "tumor",
                    "kernel_radius": 1,
                    "voxel_batch": 10000,
                    "use_torch_radiomics": use_gpu,
                    "use_gpu_matrices": use_gpu,
                    "torch_device": "auto" if use_gpu else "cpu",
                },
            ),
        )
    else:
        extract_stage = Stage(
            "extract_voxel_features",
            Spec("raw", {"modalities": ["T1", "T2"]}),
        )

    return HabitatSpec(
        name="gpu_parallel_bench",
        stages=(
            extract_stage,
            Stage(
                "preprocess1",
                Spec(
                    "winsorize",
                    {"winsor_limits": (0.05, 0.05), "across_features": False},
                ),
            ),
            Stage("preprocess2", Spec("minmax", {"across_features": False})),
            # Supervoxel search and k-means clustering.
            Stage(
                "partition",
                Spec("kmeans", {"n_supervoxels": 64, "n_init": 10}),
            ),
            Stage("pool", Spec("pool")),
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {
                        "min_habitats": 2,
                        "max_habitats": 6,
                        "validation": "elbow",
                        "n_init": 10,
                    },
                ),
            ),
            Stage("assign", Spec("nearest_centroid")),
            Stage("quantify", Spec("volume")),
            Stage("quantify2", Spec("msi")),
            Stage("quantify3", Spec("ith_score")),
        ),
        random_seed=seed,
    )


def _run_once(
    *,
    scenario: str,
    device_note: str,
    workers: int,
    cohort,
    spec,
    subject_timeout_sec: Optional[float] = None,
    cap_workers_to_gpu_pool: bool = False,
) -> BenchRow:
    """Time one Study.fit_predict under a RunPolicy."""
    from habit.execution import backend_from_policy
    from habit.recipes import Study
    from habit.spec import RunPolicy

    n_subjects = len(cohort)
    if workers <= 1:
        policy = RunPolicy(
            workers=1,
            backend="serial",
            subject_timeout_sec=subject_timeout_sec,
            on_subject_failure="fail_fast",
            resume=False,
            parallel_mode="persistent",
        )
    else:
        policy = RunPolicy(
            workers=int(workers),
            backend="process",
            subject_timeout_sec=subject_timeout_sec,
            on_subject_failure="fail_fast",
            resume=False,
            parallel_mode="persistent",
            cap_workers_to_gpu_pool=cap_workers_to_gpu_pool,
        )
    backend = backend_from_policy(policy)
    t0 = time.perf_counter()
    try:
        result = Study(spec=spec).fit_predict(cohort, backend=backend)
        wall = time.perf_counter() - t0
        n_maps = len(result.habitat_maps)
        return BenchRow(
            scenario=scenario,
            device_note=device_note,
            workers=workers,
            wall_s=wall,
            subjects_per_min=(60.0 * n_subjects / wall) if wall > 0 else float("nan"),
            peak_rss_mb=_peak_rss_mb(),
            n_maps=n_maps,
            ok=True,
        )
    except Exception as exc:  # noqa: BLE001 - report in table
        wall = time.perf_counter() - t0
        return BenchRow(
            scenario=scenario,
            device_note=device_note,
            workers=workers,
            wall_s=wall,
            subjects_per_min=0.0,
            peak_rss_mb=_peak_rss_mb(),
            n_maps=0,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _markdown_table(rows: Sequence[BenchRow]) -> str:
    """Render measured rows as a Markdown table."""
    header = (
        "| scenario | device | workers | wall_s | subjects/min | peak_RSS_MiB | "
        "n_maps | ok |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {scenario} | {device} | {workers} | {wall:.2f} | {spm:.2f} | "
            "{rss:.1f} | {n_maps} | {ok} |".format(
                scenario=row.scenario,
                device=row.device_note,
                workers=row.workers,
                wall=row.wall_s,
                spm=row.subjects_per_min,
                rss=row.peak_rss_mb,
                n_maps=row.n_maps,
                ok="yes" if row.ok else f"NO ({row.error[:40]})",
            )
        )
    return "\n".join(lines)


def run_scenario_cpu0(
    *,
    cohort,
    spec,
    worker_counts: Sequence[int],
) -> List[BenchRow]:
    """0-GPU + multi-CPU: hide all CUDA devices."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    rows: List[BenchRow] = []
    for w in worker_counts:
        print(f"[cpu0] workers={w} ...", flush=True)
        rows.append(
            _run_once(
                scenario="0gpu_multicpu",
                device_note="CUDA=-1",
                workers=w,
                cohort=cohort,
                spec=spec,
            )
        )
        print(
            f"  -> wall={rows[-1].wall_s:.2f}s ok={rows[-1].ok} {rows[-1].error}",
            flush=True,
        )
    return rows


def run_scenario_gpu1(
    *,
    cohort,
    spec,
    worker_counts: Sequence[int],
) -> List[BenchRow]:
    """1-GPU + multi-CPU: only device 0 visible."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Default single-GPU oversubscribe prefers CPU for worker>0.
    os.environ.pop("HABIT_GPU_OVERSUBSCRIBE", None)
    rows: List[BenchRow] = []
    for w in worker_counts:
        print(f"[gpu1] workers={w} ...", flush=True)
        rows.append(
            _run_once(
                scenario="1gpu_multicpu",
                device_note="CUDA=0",
                workers=w,
                cohort=cohort,
                spec=spec,
            )
        )
        print(
            f"  -> wall={rows[-1].wall_s:.2f}s ok={rows[-1].ok} {rows[-1].error}",
            flush=True,
        )
    return rows


def run_scenario_gpu_multi(
    *,
    cohort,
    spec,
    n_gpus: int,
    worker_counts: Optional[Sequence[int]] = None,
) -> List[BenchRow]:
    """Multi-GPU: expose N devices; workers map 1:1 when capped."""
    if n_gpus < 2:
        print("[gpuN] skipped (need >=2 visible GPUs)", flush=True)
        return []
    visible = ",".join(str(i) for i in range(n_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    if worker_counts is None:
        worker_counts = [1, 2, min(4, n_gpus), n_gpus]
    rows: List[BenchRow] = []
    for w in worker_counts:
        print(f"[gpuN={n_gpus}] workers={w} ...", flush=True)
        rows.append(
            _run_once(
                scenario=f"{n_gpus}gpu_multicpu",
                device_note=f"CUDA={visible}",
                workers=w,
                cohort=cohort,
                spec=spec,
                cap_workers_to_gpu_pool=(w > 1),
            )
        )
        print(
            f"  -> wall={rows[-1].wall_s:.2f}s ok={rows[-1].ok} {rows[-1].error}",
            flush=True,
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("cpu0", "gpu1", "gpuN", "all"),
        default="all",
        help="Which layout to measure (default: all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Alias for --scenario all.",
    )
    parser.add_argument("--n-subjects", type=int, default=16)
    parser.add_argument("--shape", type=_parse_shape, default=(80, 80, 48))
    parser.add_argument(
        "--workers",
        type=_parse_int_list,
        default=[1, 2, 4, 8],
        help="Worker counts for cpu0/gpu1 (default: 1,2,4,8).",
    )
    parser.add_argument(
        "--extractor",
        choices=("voxel_radiomics", "raw"),
        default="voxel_radiomics",
        help="Feature extractor: 'voxel_radiomics' (GPU-accelerated) or 'raw' (default: voxel_radiomics).",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=0,
        help="GPUs to expose for gpuN (0 = use all detected).",
    )
    parser.add_argument(
        "--workload",
        choices=("voxel_radiomics", "raw"),
        default="voxel_radiomics",
        help="Feature workload: voxel_radiomics (GPU-accelerated) or raw (CPU-bound baseline).",
    )
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args(list(argv) if argv is not None else None)
    scenario = "all" if args.all else args.scenario

    n_detected = _detect_gpus()
    print(f"Detected torch CUDA devices (before masking): {n_detected}", flush=True)
    print(
        f"Cohort: n_subjects={args.n_subjects}, shape={args.shape}, seed={args.seed}, workload={args.workload}",
        flush=True,
    )

    from habit.datasets import make_synthetic_cohort

    cohort = make_synthetic_cohort(
        n_subjects=int(args.n_subjects),
        shape=args.shape,
        rng=int(args.seed),
    )
    import numpy as np
    roi_voxels_per_subj = int(np.sum(cohort[0].mask("tumor").data > 0))
    total_roi_voxels = roi_voxels_per_subj * len(cohort)
    n_features = 90 if args.workload == "voxel_radiomics" else 2
    print(
        f"Workload scale: ~{roi_voxels_per_subj:,} tumor ROI voxels/subject, "
        f"total {total_roi_voxels:,} ROI voxels across cohort ({n_features} features/voxel).",
        flush=True,
    )
    spec_cpu = _build_spec(workload=args.workload, use_gpu=False, seed=int(args.seed))
    spec_gpu = _build_spec(workload=args.workload, use_gpu=True, seed=int(args.seed))

    rows: List[BenchRow] = []
    if scenario in ("cpu0", "all"):
        rows.extend(
            run_scenario_cpu0(
                cohort=cohort, spec=spec_cpu, worker_counts=args.workers
            )
        )
    if scenario in ("gpu1", "all"):
        rows.extend(
            run_scenario_gpu1(
                cohort=cohort, spec=spec_gpu, worker_counts=args.workers
            )
        )
    if scenario in ("gpuN", "all"):
        n_gpus = int(args.n_gpus) if args.n_gpus > 0 else max(0, n_detected)
        # Re-detect after previous scenarios may have masked devices: use host count.
        if n_gpus <= 0:
            # Force unmask briefly for counting.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            n_gpus = _detect_gpus()
        rows.extend(
            run_scenario_gpu_multi(
                cohort=cohort, spec=spec_gpu, n_gpus=n_gpus
            )
        )

    print("\n## HABIT parallel benchmark\n", flush=True)
    print(_markdown_table(rows), flush=True)

    # Speedup summary vs serial within each scenario.
    by_sc: Dict[str, List[BenchRow]] = {}
    for row in rows:
        by_sc.setdefault(row.scenario, []).append(row)
    print("\n## Speedup vs serial (same scenario)\n", flush=True)
    print("| scenario | workers | speedup_vs_w1 |", flush=True)
    print("| --- | ---: | ---: |", flush=True)
    for sc, sc_rows in by_sc.items():
        base = next((r for r in sc_rows if r.workers == 1 and r.ok), None)
        if base is None or base.wall_s <= 0:
            continue
        for r in sc_rows:
            if not r.ok:
                continue
            print(
                f"| {sc} | {r.workers} | {base.wall_s / r.wall_s:.2f}x |",
                flush=True,
            )
    return 0 if all(r.ok for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
