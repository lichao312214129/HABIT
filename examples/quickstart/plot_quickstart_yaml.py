"""
Quickstart: YAML
================

**Background.** A YAML file is a plain-text way to write the same stage list
you would build in Python, so an analysis can be run from the shell, shared,
and kept next to the results.

**Purpose.** You get the two-step habitat maps and a volume-fraction / MSI /
ITH / graph table from a YAML file, plus one habitat overlay.

**Key terms.**

* **YAML config** -- a text file whose ``spec:`` block is a
  ``HabitatSpec`` (``name``, ``random_seed``, ``stages``), and whose
  ``data:`` / ``output:`` blocks say where to read subjects and write results.
* **Stage / Spec / HabitatSpec** -- see
  :doc:`/auto_quickstart/plot_quickstart_python`; each YAML stage entry is
  one ``Stage``, and its ``component`` is the ``Spec``.

The same analysis as :doc:`plot_quickstart_python`, written as a YAML
file: same stages, same seed, fitted on the same four subjects. It gives
the same habitat maps and feature values as the Python page, and as
``habit get-habitat`` with ``config/habitat/config_habitat_quickstart_v1.yaml``
on :doc:`/tutorial/quickstart`. ``habit get-habitat --config <file>`` runs
a file like this from the shell; :func:`habit.recipes.run_from_yaml` is the
Python call behind that command. The ``spec.stages`` list is the stage
list of the Python page, one entry per stage.

Change ``DATA`` to your preprocessed root, ``SUBJECTS`` to your subject
folders, and ``MODALITIES`` / ``ROI`` to your series names. Two details
of the YAML loader:

* ``data.source`` is either a folder (every subject under it) or a
  **manifest** YAML that lists subjects and files. The manifest is how
  the four training subjects are chosen here; the fifth demo subject is
  the new patient of the Python page.
* The loader keys the ROI mask by the **first** modality, so ``LAP`` (the
  series the tumour was drawn on) is listed first. The Python page lists
  ``pre_contrast`` first; only the column order differs, the habitat maps
  do not.
"""

# %%
# Write the files and run them
# ----------------------------
# The manifest names the four subjects and their LAP masks; the YAML
# document names the stages. ``save=False`` keeps the result in memory
# instead of writing ``out_dir``.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

import habit.recipes as recipes
from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.viz import plot_habitat_overlay

# Change DATA / SUBJECTS / MODALITIES / ROI to your preprocessed layout.
DATA = Path(fetch_demo()).as_posix()
SUBJECTS = ["subj001", "subj002", "subj003", "subj004"]
MODALITIES = ["LAP", "pre_contrast", "PVP", "delay_3min"]  # ROI series first
ROI = "LAP"
Path("out").mkdir(exist_ok=True)

# Manifest: one image folder per subject and modality, one ROI mask folder
# per subject. auto_select_first_file picks the single file in each folder.
manifest_path = Path("out/quickstart_subjects.yaml")
manifest = {
    "auto_select_first_file": True,
    "images": {s: {m: f"{DATA}/images/{s}/{m}" for m in MODALITIES} for s in SUBJECTS},
    "masks": {s: {ROI: f"{DATA}/masks/{s}/{ROI}"} for s in SUBJECTS},
}
manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

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
      component: {{name: raw, params: {{modalities: [{", ".join(MODALITIES)}], roi: {ROI}}}}}
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
    - name: graph
      component: {{name: graph, params: {{include_extended_metrics: false}}}}
data:
  source: {manifest_path.as_posix()}
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
# The same columns the Python page prints; the values match that table.
table = result.features.frame.set_index("subject")
columns = [c for c in table.columns if c.endswith("_volume_fraction")] + ["ith_score", "contrast", "graph_num_nodes_total"]
print(table[columns].round(3).to_string())

# %%
# Look at one habitat map
# -----------------------
# The anatomy comes from the same folder the YAML named. The saved model
# (``save=True`` writes it under ``out_dir``) labels a new patient the
# way the Python page shows in its last section.
habitat_map = result.habitat_maps[0]
cohort = cohort_from_directory(DATA, modalities=(ROI,), roi=ROI)
subject = next(s for s in cohort if s.subject_id == habitat_map.subject_id)
fig = plot_habitat_overlay(
    subject.image(ROI),
    habitat_map,
    title=f"{subject.subject_id}: habitats (YAML)",
    crop_to="labels",
)
fig.savefig("out/quickstart_yaml_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
