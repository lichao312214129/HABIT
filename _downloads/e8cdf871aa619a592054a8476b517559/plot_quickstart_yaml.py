"""
Quickstart: YAML
================

**Background.** A YAML file is a plain-text way to write the same stage list
you would build in Python, so an analysis can be run from the shell, shared,
and kept next to the results.

**Purpose.** You get the two-step habitat maps and a volume-fraction / MSI /
ITH table from a YAML file, plus one habitat overlay.

**Key terms.**

* **YAML config** -- a text file whose ``spec:`` block is a
  ``HabitatSpec`` (``name``, ``random_seed``, ``stages``), and whose
  ``data:`` / ``output:`` blocks say where to read subjects and write results.
* **Stage / Spec / HabitatSpec** -- see
  :doc:`/auto_quickstart/plot_quickstart_python`; each YAML stage entry is
  one ``Stage``, and its ``component`` is the ``Spec``.

The same two-step analysis as :doc:`plot_quickstart_python`, written as a
YAML file. ``habit get-habitat --config <file>`` runs it from the shell;
:func:`habit.recipes.run_from_yaml` is the Python call behind that
command. The ``spec.stages`` list is the stage list of the Python page,
one entry per stage.

Change ``source`` to your preprocessed root, and the modality names to
the series folders under ``images/<subject>/``. The YAML loader reads the
ROI mask from the folder of the **first** modality, so ``LAP`` (the
series the tumour was drawn on) is listed first.
"""

# %%
# Write the file and run it
# -------------------------
# The YAML runs every subject under ``source`` (all five demo lesions),
# so the habitat count can differ from the Python page, which fits on
# four. ``save=False`` keeps the result in memory instead of writing
# ``out_dir``.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt

import habit.recipes as recipes
from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.viz import plot_habitat_overlay

DATA = Path(fetch_demo()).as_posix()
Path("out").mkdir(exist_ok=True)
yaml_path = Path("out/quickstart.yaml")
yaml_path.write_text(
    f"""\
version: '1.0'
workflow: habitat
mode: train
spec:
  name: quickstart_two_step
  random_seed: 0
  stages:
    - name: extract
      component: {{name: raw, params: {{modalities: [LAP, pre_contrast, PVP, delay_3min], roi: LAP}}}}
    - name: partition
      component: {{name: kmeans, params: {{n_supervoxels: 30}}}}
    - name: pool
      component: {{name: pool}}
    - name: fit
      component: {{name: kmeans, params: {{min_habitats: 2, max_habitats: 10, validation: elbow, n_init: 10}}}}
    - name: assign
      component: {{name: nearest_centroid}}
    - name: volume
      component: {{name: volume}}
    - name: msi
      component: {{name: msi}}
    - name: ith
      component: {{name: ith_score}}
data:
  source: {DATA}
output:
  out_dir: out/quickstart_yaml
""",
    encoding="utf-8",
)
print(yaml_path.read_text(encoding="utf-8"))

# run_from_yaml parses the file into a HabitatSpec plus a cohort and runs
# it -- the same call ``habit get-habitat`` makes.
result = recipes.run_from_yaml(yaml_path, workflow="habitat", save=False)
print(result.habitat_model.summary())
table = result.features.frame.set_index("subject")
print(table[[c for c in table.columns if c.endswith("_volume_fraction")] + ["ith_score"]].round(3).to_string())

# %%
# Look at one habitat map
# -----------------------
# The anatomy comes from the same folder the YAML named.
habitat_map = result.habitat_maps[0]
cohort = cohort_from_directory(DATA, modalities=("LAP",), roi="LAP")
subject = next(s for s in cohort if s.subject_id == habitat_map.subject_id)
fig = plot_habitat_overlay(
    subject.image("LAP"),
    habitat_map,
    title=f"{subject.subject_id}: habitats (YAML)",
    crop_to="labels",
)
fig.savefig("out/quickstart_yaml_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
