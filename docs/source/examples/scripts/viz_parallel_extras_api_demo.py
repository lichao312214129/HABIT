#!/usr/bin/env python
"""
Visualization, RunPolicy, and extras API (lightweight demo).

Full process-pool parallelism on real demo_data is exercised in
``demo_data/results/api/run_api_coverage.py`` step ``03_habitat_one_step``.
This script keeps the import/call surface short so it finishes quickly in
docs CI-like environments.

Accompanies ``docs/source/examples/viz_parallel_extras_api.rst``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from habit import HabitatSpec, RunPolicy, Spec, Stage, make_synthetic_cohort
from habit.execution.process_pool import ProcessPoolBackend
from habit.viz import (
    plot_coefficient_forest,
    plot_habitat_clustering_pca_2d,
    use_style,
)
import habit.recipes as recipes

REPO_ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    """Windows-safe entry: keep process-pool construction under ``__main__``."""
    print("=== RunPolicy object (scheduling surface) ===")
    policy = RunPolicy(
        workers=2,
        backend="process",
        on_subject_failure="continue",
        parallel_mode="isolated",
    )
    backend = ProcessPoolBackend.from_policy(policy)
    print(
        f"  workers={policy.workers}, backend={policy.backend!r}, "
        f"parallel_mode={policy.parallel_mode!r}, "
        f"ProcessPoolBackend={type(backend).__name__}"
    )

    print("=== one_step serial + habit.viz PCA ===")
    cohort = make_synthetic_cohort(n_subjects=3, shape=(12, 12, 12), rng=9)
    # Neither partition nor pool ⇒ one_step (inferred).
    spec = HabitatSpec(
        name="viz_api",
        stages=(
            Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {
                        "min_habitats": 2,
                        "max_habitats": 3,
                        "validation": "elbow",
                        "n_init": 2,
                    },
                ),
            ),
            Stage("assign", Spec("nearest_centroid")),
            Stage("quantify", Spec("volume")),
            Stage("quantify2", Spec("msi")),
            Stage("quantify3", Spec("ith_score")),
            Stage("quantify4", Spec("non_radiomics")),
        ),
        random_seed=9,
    )
    result = recipes.Study(spec=spec).fit_predict(cohort)
    # one_step has per-subject models (no cohort pipeline); atomic reuse is the
    # already-computed habitat map for that subject (or re-run one_step on a
    # one-subject cohort slice).
    one_map = result.habitat_maps[0]
    print(
        f"  habitat map {cohort[0].subject_id}: "
        f"{len(set(int(v) for v in one_map.label_array.ravel() if v > 0))} labels"
    )

    # Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _habitat_eye_check import eye_check_study
    eye_check_study(cohort, result)

    units = result.units[0]
    habitat = result.habitat_maps[0]
    unit_labels = np.asarray(units.label_array).ravel()
    habitat_labels = np.asarray(habitat.label_array).ravel()
    order = np.argsort(unit_labels, kind="stable")
    unique, first = np.unique(unit_labels[order], return_index=True)
    keep = unique != 0
    unit_ids = unique[keep].astype(np.int64)
    assigned = habitat_labels[order[first[keep]]].astype(np.int64)
    features = units.features.loc[unit_ids].to_numpy(dtype=np.float64)
    with use_style("radiology"):
        fig = plot_habitat_clustering_pca_2d(
            features, assigned, title="Habitat PCA (API demo)"
        )
        with tempfile.TemporaryDirectory(prefix="habit_viz_api_") as tmp:
            out = Path(tmp) / "pca2d.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {out.name} ({out.stat().st_size} bytes)")

    print("=== habit.viz coefficient forest ===")
    names = [f"f{i}" for i in range(6)]
    coef = np.linspace(-1.2, 1.0, num=6)
    with use_style("radiology"):
        fig = plot_coefficient_forest(names=names, coefficient=coef, title="LR coef")
        plt.close(fig)
    print(f"  forest for {len(names)} features")

    yaml_src = REPO_ROOT / "config" / "habitat" / "config_habitat_two_step_v1.yaml"
    print(
        f"=== extras: run_from_yaml / icc / retest -> "
        f"demo_data/results/api/09_extras (yaml present={yaml_src.is_file()}) ==="
    )


if __name__ == "__main__":
    main()
