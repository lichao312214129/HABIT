#!/usr/bin/env python
"""
Generate English-labelled thumbnails for the Examples gallery index.

Writes PNGs under ``docs/source/_static/images/examples/``. Safe to re-run.
Requires matplotlib (docs / viz stack).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit import HabitatSpec, Spec, Stage, make_synthetic_cohort
import habit.recipes as recipes

OUT_DIR = (
    Path(__file__).resolve().parents[2] / "_static" / "images" / "examples"
)


def main() -> None:
    """Fit a tiny two-step study and save mid-slice habitat overlays."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = make_synthetic_cohort(
        n_subjects=3,
        modalities=("T1", "T2"),
        shape=(24, 24, 24),
        rng=42,
    )
    spec = HabitatSpec(
        name="thumb_two_step",
        stages=(
            Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
            Stage("partition", Spec("kmeans", {"n_supervoxels": 8, "n_init": 3})),
            Stage("pool", Spec("pool")),
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {
                        "min_habitats": 2,
                        "max_habitats": 3,
                        "validation": "silhouette",
                        "n_init": 3,
                    },
                ),
            ),
            Stage("assign", Spec("nearest_centroid")),
            Stage("quantify", Spec("volume")),
        ),
        random_seed=42,
    )
    result = recipes.fit_habitat(cohort, spec)
    subject = cohort[0]
    anatomy = subject.image("T1").data
    labels = result.habitat_maps[0].label_array
    z = anatomy.shape[0] // 2

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=120)
    axes[0].imshow(anatomy[z], cmap="gray")
    axes[0].set_title("Anatomy (T1 mid-slice)")
    axes[0].axis("off")
    axes[1].imshow(anatomy[z], cmap="gray")
    masked = np.ma.masked_where(labels[z] == 0, labels[z])
    axes[1].imshow(masked, cmap="tab10", alpha=0.55, interpolation="nearest")
    axes[1].set_title("Habitats overlay")
    axes[1].axis("off")
    fig.tight_layout()
    out = OUT_DIR / "habitat_two_step_overlay.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
