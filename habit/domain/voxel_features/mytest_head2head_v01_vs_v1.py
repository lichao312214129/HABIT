#!/usr/bin/env python
"""Head-to-head: v0.1 compat engine vs v1 recipe — direct_pooling voxel texture.

Identical pipeline on both sides (aligned with the v0.1 pooling YAML defaults):

    voxel_radiomics(T1, kernel_radius=3, bundled R3B12 preset, torch auto)
      -> winsorize(0.05, 0.05) -> minmax                     (per-subject)
      -> winsorize -> variance_filter(0.01)
         -> correlation_filter(0.9, spearman) -> minmax      (cohort level)
      -> kmeans, k in 2..10, elbow selection, n_init=10, seed=42

Four legs, each timed wall-clock (compute + built-in result writing, i.e. what
a CLI user actually waits for):

    v01_serial     v0.1 engine, processes=1
    v01_parallel   v0.1 engine, processes=2 (as-shipped defaults: persistent,
                   auto_retry_rounds=2, no GPU cap)
    v1_serial      recipes.direct_pooling, SerialBackend
    v1_parallel    recipes.direct_pooling, ProcessPoolBackend workers=2
                   (persistent, auto_retry_rounds=0, no GPU cap — same worker
                   count and same shared-GPU conditions as the v0.1 leg)

Each leg runs as a fresh subprocess (cold interpreter + cold CUDA context for
both sides, symmetric import cost — honest CLI wall time) and writes a small
``timing.json`` into its output directory. ``compare`` then checks the habitat
label maps voxel-by-voxel (raw agreement + agreement after optimal label
permutation, since kmeans label ids are arbitrary) and the per-subject habitat
size distributions (label-agnostic).

Usage::

    python habit/domain/voxel_features/mytest_head2head_v01_vs_v1.py all
    python habit/domain/voxel_features/mytest_head2head_v01_vs_v1.py compare

Env overrides: HABIT_H2H_N (subjects, default 5).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# Cap BLAS/OpenMP threads for BOTH sides equally (this script is the parent of
# every leg; spawned pool workers inherit these). Without the cap, N workers x
# default BLAS threads oversubscribe the CPU and distort the comparison.
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
# Run as ``python <this file>`` puts the script's own directory on sys.path,
# so ``import habit`` would silently resolve to the *installed* copy in
# site-packages (1.0.1 at the time of writing) instead of this working tree.
# Pin the repo first: the whole point of the head-to-head is measuring the
# current code (e.g. the _AssignPrecomputedUnits units-reuse path).
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / ".cursor" / "test" / "resample_02"
V01_YAML = REPO_ROOT / "config" / "habitat" / "config_habitat_pooling_voxel_radiomics_train.yaml"
OUT_ROOT = REPO_ROOT / "out" / "head2head"
SUBSET = OUT_ROOT / "data_t1"  # v0.1-layout subset: images/<sub>/T1/, masks/<sub>/T1/
N_SUBJECTS = int(os.environ.get("HABIT_H2H_N", "5"))
MODALITY = "T1"

LEGS = ("v01_serial", "v01_parallel", "v1_serial", "v1_parallel")


def prepare_data() -> List[str]:
    """Copy the first N subjects (T1 image + mask) into a v0.1-layout subset.

    Args:
        None (uses module-level DATA / SUBSET / N_SUBJECTS).

    Returns:
        List[str]: subject ids included, in lexicographic order (the same
        order ``cohort_from_directory`` yields them on the v1 side).
    """
    images_root = DATA / "images"
    subjects = sorted(p.name for p in images_root.iterdir() if p.is_dir())[:N_SUBJECTS]
    for sub in subjects:
        for kind, filename in (("images", f"{MODALITY}.nii.gz"), ("masks", f"mask_{MODALITY}.nii.gz")):
            src = DATA / kind / sub / MODALITY / filename
            dst = SUBSET / kind / sub / MODALITY / filename
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    print(f"[prepare] {len(subjects)} subjects -> {SUBSET}: {subjects}", flush=True)
    return subjects


def run_v01_leg(leg: str, processes: int) -> None:
    """Run one v0.1 engine leg (train mode) and write timing.json.

    Args:
        leg: Leg name; also the output subdirectory under OUT_ROOT.
        processes: v0.1 ``processes`` setting (1 = serial, 2 = parallel).
    """
    from habit.schemas.workflows.habitat import HabitatAnalysisConfig
    from habit.utils.config_loader import load_config
    from habit.compat.engines.habitat_analysis.run import run_habitat_analysis_from_config

    out_dir = OUT_ROOT / leg
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(V01_YAML))
    cfg["data_dir"] = str(SUBSET)
    cfg["out_dir"] = str(out_dir)
    cfg["feature_construction"]["voxel_level"]["method"] = f"concat(voxel_radiomics({MODALITY}))"
    cfg["processes"] = processes
    cfg["resume"] = False  # timing must never read a checkpoint cache
    cfg["plot_curves"] = False
    cfg["save_images"] = True
    cfg["save_results_csv"] = True
    # As-shipped v0.1 defaults kept on purpose: persistent mode,
    # individual_subject_auto_retry_rounds=2, cap_processes_to_gpu_pool=false.
    config = HabitatAnalysisConfig.from_dict(cfg)

    t0 = time.perf_counter()
    run_habitat_analysis_from_config(config)
    wall = time.perf_counter() - t0
    (out_dir / "timing.json").write_text(json.dumps({"leg": leg, "wall_s": wall}))
    print(f"[{leg}] wall {wall:.2f}s -> {out_dir}", flush=True)


def run_v1_leg(leg: str, parallel: bool) -> None:
    """Run one v1 recipe leg and write timing.json.

    Args:
        leg: Leg name; also the save subdirectory under OUT_ROOT.
        parallel: When True, use a persistent ProcessPoolBackend with 2
            workers; otherwise the in-process SerialBackend.
    """
    from habit import HabitatSpec, RunPolicy, Spec, cohort_from_directory
    import habit.recipes as recipes

    cohort = cohort_from_directory(SUBSET, modalities=(MODALITY,), roi=MODALITY)
    print(f"[{leg}] cohort={len(tuple(cohort))} subjects", flush=True)

    spec = HabitatSpec(
        name=f"h2h_{leg}",
        voxel_feature_extractor=Spec(
            "voxel_radiomics",
            {"modalities": [MODALITY], "kernel_radius": 3, "use_torch_radiomics": True},
        ),
        voxel_feature_preprocessors=(
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
            Spec("minmax", {"across_features": False}),
        ),
        supervoxelizer=None,  # direct pooling: every ROI voxel is a unit
        cohort_feature_preprocessors=(
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
            # Column filters act per feature column by definition; they take
            # no ``across_features`` switch (unlike scalers).
            Spec("variance_filter", {"variance_threshold": 0.01}),
            Spec("correlation_filter", {"corr_threshold": 0.9, "corr_method": "spearman"}),
            Spec("minmax", {"across_features": False}),
        ),
        habitat_model_fitter=Spec(
            "kmeans",
            # v0.1 YAML default: k in 2..10, elbow, n_init=10, max_iter=300.
            {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10},
        ),
        habitat_assigner=Spec("nearest_centroid"),
        habitat_features=(Spec("volume"),),
        random_seed=42,
    )

    backend = None
    if parallel:
        from habit.execution.process_pool import ProcessPoolBackend

        backend = ProcessPoolBackend.from_policy(
            RunPolicy(
                workers=2,
                backend="process",
                on_subject_failure="continue",
                parallel_mode="persistent",
                # Same as the v0.1 leg: 2 workers share the single GPU.
                cap_workers_to_gpu_pool=False,
                auto_retry_rounds=0,
            )
        )

    t0 = time.perf_counter()
    result = recipes.direct_pooling(cohort, spec, backend=backend)
    save_dir = result.save(
        str(OUT_ROOT / leg),
        map_format="nrrd",
        write_cluster_plots=False,
        write_cluster_plots_3d=False,
        write_interactive_cluster_plots=False,
    )
    wall = time.perf_counter() - t0
    (Path(save_dir) / "timing.json").write_text(
        json.dumps({"leg": leg, "wall_s": wall, "n_habitats": result.habitat_model.n_habitats})
    )
    print(
        f"[{leg}] wall {wall:.2f}s | habitats={result.habitat_model.n_habitats} -> {save_dir}",
        flush=True,
    )


def compare() -> None:
    """Compare timing + habitat maps + habitat sizes across all four legs."""
    import numpy as np
    import SimpleITK as sitk
    from scipy.optimize import linear_sum_assignment

    print("\n===== wall time (compute + result writing) =====")
    for leg in LEGS:
        path = OUT_ROOT / leg / "timing.json"
        if path.exists():
            info = json.loads(path.read_text())
            extra = f", habitats={info['n_habitats']}" if "n_habitats" in info else ""
            print(f"  {leg:14s}: {info['wall_s']:8.2f} s{extra}")
        else:
            print(f"  {leg:14s}: (missing)")

    def load_maps(leg: str) -> dict:
        """Return {subject: label ndarray} for one leg's output directory."""
        maps = {}
        for f in sorted((OUT_ROOT / leg).glob("*_habitats.nrrd")):
            maps[f.name.replace("_habitats.nrrd", "")] = sitk.GetArrayFromImage(sitk.ReadImage(str(f)))
        return maps

    def best_permutation_agreement(a: np.ndarray, b: np.ndarray) -> float:
        """Voxel agreement after the optimal 1:1 habitat-label relabeling.

        KMeans label ids are arbitrary, so two identical partitions can still
        disagree on raw ids; the Hungarian assignment on the confusion matrix
        recovers the best label correspondence.
        """
        mask = (a > 0) | (b > 0)  # union of ROI voxels
        av, bv = a[mask].ravel(), b[mask].ravel()
        labels = sorted(set(av.tolist()) | set(bv.tolist()))
        idx = {lab: i for i, lab in enumerate(labels)}
        conf = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for x, y in zip(av, bv):
            conf[idx[x], idx[y]] += 1
        row, col = linear_sum_assignment(-conf)
        return conf[row, col].sum() / mask.sum()

    ref = load_maps("v01_serial")
    print("\n===== habitat maps vs v01_serial (voxel agreement) =====")
    for leg in ("v01_parallel", "v1_serial", "v1_parallel"):
        other = load_maps(leg)
        common = sorted(set(ref) & set(other))
        if not common:
            print(f"  {leg:14s}: no common subjects (ref={sorted(ref)}, other={sorted(other)})")
            continue
        raw_vals, perm_vals, ks = [], [], []
        for sub in common:
            a, b = ref[sub], other[sub]
            if a.shape != b.shape:
                print(f"  {leg:14s}: {sub} shape mismatch {a.shape} vs {b.shape}")
                continue
            raw_vals.append(float((a == b).mean()))
            perm_vals.append(best_permutation_agreement(a, b))
            ks.append(int(b.max()))
        if raw_vals:
            print(
                f"  {leg:14s}: raw={np.mean(raw_vals) * 100:.2f}%  "
                f"permuted={np.mean(perm_vals) * 100:.2f}%  "
                f"k={sorted(set(ks))} (v01 k={sorted({int(ref[s].max()) for s in common})})"
            )

    print("\n===== per-subject habitat sizes (sorted voxel counts, label-agnostic) =====")
    for leg in LEGS:
        maps = load_maps(leg)
        for sub in sorted(maps)[:2]:  # first two subjects are enough to eyeball drift
            arr = maps[sub]
            counts = sorted((int((arr == k).sum()) for k in range(1, int(arr.max()) + 1)), reverse=True)
            print(f"  {leg:14s} {sub}: {counts}")


def main() -> None:
    """Dispatch legs; ``all`` runs each leg as a fresh subprocess."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    mode = sys.argv[1]
    if mode == "prepare":
        prepare_data()
    elif mode == "all":
        prepare_data()
        for leg in LEGS:
            print(f"\n########## {leg} ##########", flush=True)
            subprocess.run([sys.executable, str(Path(__file__)), leg], cwd=REPO_ROOT, check=True)
        compare()
    elif mode == "compare":
        compare()
    elif mode == "v01_serial":
        run_v01_leg("v01_serial", processes=1)
    elif mode == "v01_parallel":
        run_v01_leg("v01_parallel", processes=2)
    elif mode == "v1_serial":
        run_v1_leg("v1_serial", parallel=False)
    elif mode == "v1_parallel":
        run_v1_leg("v1_parallel", parallel=True)
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
