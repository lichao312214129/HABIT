#!/usr/bin/env python
"""
Discover plugins and document the third-party entry-point pattern.

In-process registration is shown in ``custom_voxel_feature_demo.py``.
This script focuses on discovery (:func:`habit.list_plugins`) and loading
(:func:`habit.load_plugins`) so a separate package can expose components via::

    [project.entry-points."habit.voxel_feature_extractor"]
    my_feature = "my_pkg.features:register"

where ``register()`` performs ``@VoxelFeatureExtractorRegistry.register(...)``.

Accompanies ``docs/source/examples/plugin_entry_points.rst``.

Run from the repository root::

    python docs/source/examples/scripts/plugin_entry_points_demo.py
"""

from __future__ import annotations

from habit import list_plugins, load_plugins
from habit.domain import VoxelFeatureExtractorRegistry


def main() -> None:
    """Load entry points (if any) and print registered voxel extractors."""
    report = load_plugins(strict=False)
    print(
        f"load_plugins: loaded={len(report.loaded)}, "
        f"failures={len(report.failures)}"
    )
    if report.failures:
        for name, err in list(report.failures.items())[:5]:
            print(f"  failed: {name}: {err}")

    infos = list_plugins("voxel_feature_extractor")
    print(f"list_plugins('voxel_feature_extractor'): {len(infos)} entries")
    for info in infos[:12]:
        print(f"  - {info.name}")

    # Registry names after load (built-ins always present).
    names = sorted(VoxelFeatureExtractorRegistry.available())
    print(f"VoxelFeatureExtractorRegistry.available(): {names}")
    print(
        "Third-party packages: declare habit.<domain> entry points, "
        "call load_plugins(), then Spec('your_name', {...}) in HabitatSpec."
    )

    from pathlib import Path

    from habit import make_synthetic_cohort, one_step_habitat
    from habit.viz import plot_habitat_overlay

    cohort = make_synthetic_cohort(n_subjects=2, modalities=("T1", "T2"), rng=0)
    result = one_step_habitat(
        modalities=("T1", "T2"), n_habitats=3, random_seed=0
    ).fit_predict(cohort)
    fig = plot_habitat_overlay(
        cohort[0].image("T1").data,
        result.habitat_maps[0].label_array,
        axis=0,
        title="After load_plugins: one-step habitats",
    )
    Path("out").mkdir(exist_ok=True)
    fig.savefig("out/plugin_entry_points_overlay.png", dpi=150, bbox_inches="tight")
    print("Wrote out/plugin_entry_points_overlay.png")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_example_figure

    save_example_figure(fig, "plugin_entry_points_overlay.png")


if __name__ == "__main__":
    main()
