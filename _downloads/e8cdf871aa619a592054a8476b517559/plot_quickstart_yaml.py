"""
Load a habitat analysis from YAML
=================================

:func:`habit.recipes.run_from_yaml` is the Python call behind
``habit get-habitat --config``. The document below is a v1 file:
``version``, ``workflow``, ``spec`` (the stage list), ``data``, and
``policy``. Paths in ``data.source`` are used as written.

Change ``DATA`` to your preprocessed root, and the modality names to
the series folders under ``images/<subject>/``. The first modality is
the ROI mask folder this loader uses.
"""

# %%
# Write a short two-step document next to the script output, then run it.
# ``save=False`` keeps the :class:`~habit.recipes.StudyResult` in memory.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

from habit.datasets import fetch_demo
from habit.viz import plot_habitat_overlay
import habit.recipes as recipes

DATA = fetch_demo()
MODALITIES = ("LAP", "PVP")
OUT = Path("out")
OUT.mkdir(exist_ok=True)
yaml_path = OUT / "quickstart_two_step.yaml"
yaml_path.write_text(
    "\n".join(
        [
            "version: '1.0'",
            "workflow: habitat",
            "mode: train",
            "spec:",
            "  name: habitat_two_step",
            "  stages:",
            "    - name: extract_voxel_features",
            "      component:",
            "        name: raw",
            "        params:",
            "          modalities: [" + ", ".join(MODALITIES) + "]",
            "    - name: partition",
            "      component:",
            "        name: kmeans",
            "        params:",
            "          n_supervoxels: 12",
            "          n_init: 3",
            "    - name: pool",
            "      component:",
            "        name: pool",
            "    - name: fit",
            "      component:",
            "        name: kmeans",
            "        params:",
            "          n_habitats: 3",
            "          n_init: 3",
            "    - name: assign",
            "      component:",
            "        name: nearest_centroid",
            "    - name: quantify",
            "      component:",
            "        name: volume",
            "  random_seed: 0",
            "data:",
            f"  source: {Path(DATA).as_posix()}",
            "policy:",
            "  workers: 1",
            "  backend: serial",
            "  resume: false",
            "output:",
            "  out_dir: out/quickstart_yaml",
            "  save_images: false",
            "  plot_curves: false",
            "",
        ]
    ),
    encoding="utf-8",
)
print(yaml_path.read_text(encoding="utf-8"))

result = recipes.run_from_yaml(yaml_path, workflow="habitat", save=False)
print(result.habitat_model.summary())
subject_id = result.habitat_maps[0].subject_id
print("labelled:", [habitat_map.subject_id for habitat_map in result.habitat_maps])

# %%
# Overlay the first subject's map. Anatomy is loaded from the same root
# the YAML named.
from habit.contracts import cohort_from_directory

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=MODALITIES[0])
subject = next(item for item in cohort if item.subject_id == subject_id)
fig = plot_habitat_overlay(
    subject.image(MODALITIES[0]),
    result.habitat_maps[0],
    title="habitats",
)
fig.savefig(OUT / "quickstart_yaml_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
