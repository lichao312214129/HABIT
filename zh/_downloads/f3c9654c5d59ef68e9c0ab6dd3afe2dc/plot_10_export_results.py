"""
Export results and draft the methods paragraph
==============================================

**Background.** A finished habitat run leaves three things a paper or
a collaborator needs: the habitat maps, a one-row-per-patient feature
table, and a written description of what actually ran. HABIT stores the
stage list, seeds and software versions in the run manifest; 
``describe_methods()`` turns that record into a draft methods paragraph.
``StudyResult.save`` writes the conventional on-disk layout so nothing
is left only in memory.

**Purpose.** You will fit the approved two-step analysis on two demo
patients, print the methods draft and the Spec fingerprint, save the
study (model, maps, feature table, manifest), list what landed on disk,
and write a CSV a statistics tool can open.

**When to use.** At the end of every analysis you intend to report or
share: before the manuscript methods section, before handing files to a
statistician, and before archiving a run.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **analysis definition** -- the approved two-step stage list of
  :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`.
* **RunManifest** -- the provenance record of one ``fit_predict`` /
  ``predict`` call: stages that ran, parameters, seeds, versions.
* **describe_methods()** -- draft English methods text from the
  manifest (styles include ``radiology``); edit it for your journal.
* **fingerprint** -- short digest of the Spec; two Specs with the same
  fingerprint declare the same analysis.
* **StudyResult.save** -- writes ``habitat_model.habitatmodel``, habitat
  NRRD maps, the feature table and the manifest under one directory.

.. note::

   Two demo patients are enough to show the export layout. The methods
   paragraph is a draft from what ran on this machine; you still edit
   it for inclusion criteria, scanners and clinical endpoints.

Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed tree.
"""

# %%
# Fit the approved analysis
# -------------------------
# Same stage list as the reference page. Two patients define the
# habitats; volume / MSI / ITH / graph fill the feature table that will
# be exported.
# sphinx_gallery_thumbnail_number = 1
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import HabitatModel, cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import backend_from_policy
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.policy import RunPolicy
from habit.viz import plot_habitat_overlay

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train = cohort[:2]
print(train)
Path("out").mkdir(exist_ok=True)

spec = HabitatSpec(
    name="export_methods_two_step",
    stages=(
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
        Stage("preprocess2", Spec("zscore")),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
        Stage("pool", Spec("pool")),

        Stage(
            "fit",
            Spec(
                "kmeans",
                {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10},
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("volume", Spec("volume")),
        Stage("msi", Spec("msi")),
        Stage("ith", Spec("ith_score")),
        Stage("graph", Spec("graph", {"include_extended_metrics": False})),
    ),
    random_seed=0,
)
result = Study(spec).fit_predict(
    train,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
print(result.habitat_model.summary())

# %%
# Elbow / Kneedle caveat
# ----------------------
# HABIT validation ``elbow`` is the same rule as ``kneedle`` (Satopaa et
# al., 2011): KneeLocator on inertia, curve convex, direction decreasing —
# not Thorndike (1953) informal elbow, not pre-v1.0 discrete-curvature.
# Maps keep the Kneedle K. See :doc:`/user_guide/building_habitat_maps`.
_report = result.habitat_model.preprocessing_state["selection_report"]
_cand = [int(k) for k in _report["candidates"]]
_iner = np.asarray(_report["scores"][_report["methods"][0]], dtype=float)
_k_kn = int(result.habitat_model.n_habitats)
_k_dc = int(_cand[int(np.argmax(np.diff(_iner, n=2))) + 1]) if len(_iner) >= 3 else _k_kn
from habit.habitat_model import KMeansHabitatModelFitter as _KF

_k_table = {"elbow_kneedle": _k_kn, "discrete_curvature_elbow": _k_dc}
for _name in ("silhouette", "calinski_harabasz", "davies_bouldin"):
    _f = _KF(min_habitats=2, max_habitats=10, validation=_name, n_init=10)
    _f.set_random_state(0)
    _k_table[_name] = _f.fit(result.units, cohort=train).n_habitats
print("K by criterion (skip gap):")
for _n, _k in _k_table.items():
    print(f"  {_n}: {_k}")

fig = plot_habitat_overlay(
    train[0].image("LAP"),
    result.habitat_maps[0],
    title=f"{train[0].subject_id}: habitats",
    crop_to="labels",
)
fig.savefig("out/export_methods_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Spec fingerprint and methods draft
# ----------------------------------
# The fingerprint is a digest of the declared Spec. The methods text
# comes from the **run** manifest: only stages that actually executed,
# with the parameters and seeds that were used. Print both styles so
# you can pick the tone closest to your journal.
print("spec fingerprint:", spec.fingerprint())
print("\n--- describe_methods (radiology) ---\n")
print(result.manifest.describe_methods(style="radiology"))
print("\n--- describe_methods (default) ---\n")
print(result.manifest.describe_methods())
versions = dict(result.manifest.software_versions())
print("software versions (sample):", {k: versions[k] for k in list(versions)[:6]})
print("random seeds:", dict(result.manifest.random_seeds()))

# %%
# Save the study directory
# ------------------------
# ``result.save`` writes the conventional layout: the ``.habitatmodel``
# archive, habitat maps, the feature table and the run manifest. Set
# ``write_maps=False`` if you only need the table and the model.
out_dir = Path("out/export_methods_run")
result.save(out_dir, write_maps=True, table_format="csv")
written = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())
print("wrote", len(written), "files under", out_dir)
for path in written:
    print(" ", path)

# %%
# Reload the model and reread the feature table
# ---------------------------------------------
# A short round-trip check: the saved model reloads, and the CSV row
# count matches the in-memory feature table.
model_path = out_dir / "habitat_model.habitatmodel"
model = HabitatModel.load(model_path)
print("reloaded model id:", model.model_id)
print("same centroids:", np.array_equal(model.centroids, result.habitat_model.centroids))

# Per-patient feature table written by ``save`` (not ``habitats.csv``,
# which is the per-supervoxel units table).
table_path = out_dir / "habitat_features.csv"
if not table_path.is_file():
    table_path = Path("out/export_methods_features.csv")
    result.features.frame.to_csv(table_path, index=False)
table = pd.read_csv(table_path)
print("feature table:", table_path.name, table.shape)
# Print a short preview: subject + volume fractions + ITH (full CSV stays on disk).
preview_cols = ["subject"] + [
    c for c in table.columns if c.endswith("_volume_fraction") or c == "ith_score"
]
print(table[preview_cols].round(3).to_string(index=False))

# Explicit CSV next to the study directory, for tools that expect a
# single flat file without browsing the run folder.
flat_csv = Path("out/export_methods_features.csv")
result.features.frame.to_csv(flat_csv, index=False)
print("flat CSV:", flat_csv, flat_csv.stat().st_size, "bytes")

# %%
# Manifest JSON (what the archive remembers)
# ------------------------------------------
# The ``.habitatmodel`` is a ZIP. Its ``manifest.json`` holds the format
# version, model id, feature columns, Spec payload and cohort fingerprint
# without identifying patient data beyond a digest of subject ids.
import zipfile

with zipfile.ZipFile(model_path) as archive:
    manifest = json.loads(archive.read("manifest.json"))
print("format_version:", manifest.get("format_version"))
print("n_habitats:", manifest.get("n_habitats") or model.n_habitats)
print("feature_names:", manifest.get("feature_names") or model.feature_names)
print("cohort subjects:", model.cohort_fingerprint.n_subjects)

# %%
# How to read the result
# ----------------------
# Measured on this run (K = 4 by elbow on subj001 + subj002):
#
# * ``describe_methods()`` drafted a methods paragraph naming every
#   stage that ran (winsorize / z-score / 30 k-means supervoxels /
#   10-bin cohort binning / k-means elbow / nearest centroid / volume /
#   MSI / ITH / graph), seed 0, HABIT 3.0.0 and the two subjects.
# * ``result.save`` wrote 8 files: ``habitat_model.habitatmodel``,
#   ``habitat_features.csv`` (2 rows x hundreds of columns),
#   ``habitats.csv`` (per-supervoxel units), ``run_manifest.json``, and
#   NRRD habitat + supervoxel maps for both patients.
# * Reloading the ``.habitatmodel`` recovered the same model id and
#   bit-identical centroids.
#
# Edit the draft methods text for your journal (scanners, inclusion
# criteria, clinical endpoints). The fingerprint identifies the Spec;
# the manifest identifies what actually ran.

# %%
# Where to go next
# ----------------
# * Train / save / predict with a held-out split:
#   :doc:`/auto_examples/05_validation_and_reuse/plot_01_train_save_predict`.
# * Hand a published ``.habitatmodel`` to another site:
#   :doc:`/auto_examples/05_validation_and_reuse/plot_06_reuse_published_model`.
# * The reference analysis this export came from:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`.
# * Per-habitat radiomics columns in the same table:
#   :doc:`/auto_examples/04_quantifying_habitats/plot_05_habitat_radiomics`.
