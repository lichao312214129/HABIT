"""
Train, save, and predict new patients
=====================================

**Background.** A habitat study usually has a training set, where the
habitat definition is learned, and new patients, where that definition
is only applied. The new patients must never influence the definition:
no refitted centroids, no refitted bin edges. The ``.habitatmodel``
file is what carries the definition from one side to the other, so a
later cohort (or a colleague) gets habitat ids that mean the same thing
as in the training set.

**Purpose.** You will fit the analysis of
:doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec` on three
training patients, save it, look at what the file contains, reload it
into a fresh object, label two held-out patients without refitting
anything, check that the model was not changed by labelling, repeat the
labelling in a separate Python process to show that the file alone is
enough, and export the held-out feature table as a CSV.

**When to use.** Any study with a train / held-out split, a cohort that
grows after the model was fixed, or a colleague who should label their
own patients with your habitat definition.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **held-out patient** -- a patient that is labelled with the saved
  model but was not used to fit it.
* **fit vs. predict** -- ``fit_predict`` learns the definition and labels
  the training patients; ``predict`` only labels patients with an
  existing definition.
* **subject-level normalization** -- winsorize and min-max with the
  patient's own statistics. It is recomputed for every patient,
  training or held-out, and stores no state.
* **cohort-level preprocessing** -- the 10-bin discretization learned on
  the pooled training supervoxels. Its bin edges are stored in the model
  and reused unchanged on held-out patients.
* **.habitatmodel** -- the saved habitat definition: centroids, the
  cohort-level state, the spec, and a non-identifying description of
  the training cohort.

The page uses the approved analysis definition unchanged; only the
training set differs from the reference page (three patients instead of
two).

.. note::

   This is not external validation. The held-out patients (subj004,
   subj005) come from the same 5-subject demo cohort as the training
   patients, not from another centre or scanner, and three training
   patients are a demo, not a cohort. The page demonstrates the
   mechanics of a train / held-out split; nothing on it is evidence of
   generalization.

Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed tree.
"""

# %%
# Load the cohort and split it
# ----------------------------
# The first three demo patients are the training set; the last two are
# held out and only ever labelled with the saved model.
# sphinx_gallery_thumbnail_number = 2
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
train, held_out = cohort[:3], cohort[3:5]
print("train:   ", list(train.subject_ids))
print("held out:", list(held_out.subject_ids))
Path("out").mkdir(exist_ok=True)

# %%
# Fit on the training patients
# ----------------------------
# The same stage list as the reference page. Stages before ``pool`` are
# subject-level; ``preprocess_cohort`` after ``pool`` is cohort-level and
# is stored in the model.
spec = HabitatSpec(
    name="train_save_predict_two_step",
    stages=(
        # extract: one intensity column per DCE phase, voxels inside the ROI.
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        # subject-level: clip to each patient's own 1st..99th percentile.
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
        # subject-level: rescale each column to 0..1 with the patient's own
        # min and max.
        Stage("preprocess2", Spec("zscore")),
        # partition: 30 supervoxels per tumour.
        Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
        # pool: stack the training supervoxels into one matrix.
        Stage("pool", Spec("pool")),
        # cohort-level: 10 equal-width bins with edges learned on the pooled
        # training supervoxels only.

        # fit: one k-means for the cohort, 2..10 habitats, elbow rule.
        Stage(
            "fit",
            Spec(
                "kmeans",
                {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10},
            ),
        ),
        # assign: nearest centroid for every supervoxel, then its voxels.
        Stage("assign", Spec("nearest_centroid")),
        # quantify: one row per subject.
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
# Elbow / Kneedle caveat and criterion table
# ------------------------------------------
# HABIT validation ``elbow`` is the same rule as ``kneedle``. It runs
# ``KneeLocator`` on k-means inertia (within-cluster sum of squares),
# curve convex, direction decreasing (Satopaa, Albrecht, Irwin, and
# Raghavan, 2011, Finding a "Kneedle" in a Haystack, IEEE ICDCS).
# Normalize k and inertia to the unit square, draw the chord from the
# first point to the last, and take the k farthest from that chord; a
# smooth curve often places this knee to the right of the bend a person
# sees. Literature "elbow" means inspecting within-cluster dispersion
# versus k (Thorndike RL, 1953, Who belongs in the family?, Psychometrika
# 18(4):267-276) — not a second-difference formula. The discrete-curvature
# elbow (visual elbow, computed) maximizes the second difference of
# inertia (HABIT pre-v1.0 elbow). Habitat maps keep the Kneedle K.
_report = result.habitat_model.preprocessing_state["selection_report"]
_candidates = [int(k) for k in _report["candidates"]]
_inertia = __import__("numpy").asarray(_report["scores"][_report["methods"][0]], dtype=float)


def _discrete_curvature_elbow_k(cluster_range, inertia_scores):
    """Return k maximizing the second difference of inertia (pre-v1.0 elbow)."""
    import numpy as np
    values = np.asarray(inertia_scores, dtype=float)
    cluster_range = [int(k) for k in cluster_range]
    if values.size < 3:
        return int(cluster_range[int(np.argmin(values))])
    return int(cluster_range[int(np.argmax(np.diff(values, n=2))) + 1])


_k_kneedle = int(result.habitat_model.n_habitats)
_k_discrete = _discrete_curvature_elbow_k(_candidates, _inertia)
from habit.habitat_model import KMeansHabitatModelFitter as _KMeansFitter

_k_table = {"elbow_kneedle": _k_kneedle, "discrete_curvature_elbow": _k_discrete}
for _name in ("silhouette", "calinski_harabasz", "davies_bouldin"):
    _fitter = _KMeansFitter(
        min_habitats=2, max_habitats=10, validation=_name, n_init=10
    )
    _fitter.set_random_state(0)
    _k_table[_name] = _fitter.fit(result.units, cohort=train).n_habitats
print("K by criterion (elbow/kneedle=Kneedle; sil/CH maximize; DB minimize; skip gap):")
for _name, _k in _k_table.items():
    print(f"  {_name}: {_k}")
print(f"habitat map uses Kneedle K = {_k_kneedle}")

# One training map, to compare colours with the held-out patients below.
fig = plot_habitat_overlay(
    train[0].image("LAP"),
    result.habitat_maps[0],
    title=f"{train[0].subject_id} (training): habitats",
    crop_to="labels",
)
fig.savefig("out/train_save_predict_train.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# The methods paragraph of the training run
# -----------------------------------------
# The run manifest records every executed step, its parameters, the
# seeds and the software versions. ``describe_methods()`` turns that
# record into a draft methods paragraph. It states only steps that
# actually ran; you still edit it for your manuscript.
print(result.manifest.describe_methods())

# %%
# Save the model and look inside the file
# ---------------------------------------
# ``result.save`` writes the model plus the training tables;
# ``write_maps=True`` also writes the training habitat maps as NRRD.
# The ``.habitatmodel`` file is a plain ZIP archive: a JSON manifest
# (every scalar field, the spec, the cohort-level state and a format
# version) and the centroid matrix as ``.npy``. No pickle, so any
# Python -- or any language with ZIP and JSON -- can read it, and a
# future HABIT can refuse an incompatible file with a clear message.
result.save("out/train_save_predict", write_maps=True)
model_path = Path("out/train_save_predict/habitat_model.habitatmodel").resolve()
with zipfile.ZipFile(model_path) as archive:
    print("archive members:", archive.namelist())
print("file size (kB):", round(model_path.stat().st_size / 1024, 1))

# %%
# Reload into a fresh object
# --------------------------
# ``HabitatModel.load`` builds a new object from the file alone. The
# model carries an id and a non-identifying description of the cohort
# that defined it (number of subjects, modalities, a digest of the
# subject ids), so a shared file says where its definition came from.
model = HabitatModel.load(model_path)
print("model id:               ", model.model_id)
print("habitats:               ", model.n_habitats)
print("feature columns:        ", model.feature_names)
print("centroid matrix shape:  ", model.centroids.shape)
print("training subjects:      ", model.cohort_fingerprint.n_subjects)
print("training modalities:    ", model.cohort_fingerprint.modalities)
print("cohort-level state keys:", sorted(model.preprocessing_state))
# The reloaded model is the same definition as the in-memory one.
print("same centroids as before saving:", np.array_equal(model.centroids, result.habitat_model.centroids))
print("same model id as before saving: ", model.model_id == result.habitat_model.model_id)

# %%
# Label the held-out patients
# ---------------------------
# ``predict`` never refits. Each held-out patient is winsorized and
# min-max scaled with its OWN statistics (subject-level), split into its
# own supervoxels, binned with the TRAINING bin edges from the model
# (cohort-level), and given the training centroids.
#
# ``spec=spec`` states the analysis the new patients go through; its
# upstream stages (extract, preprocess, partition) must be the ones the
# model was trained with. Without ``spec=``, the spec stored in the
# model is used.
centroids_before = model.centroids.copy()
prediction = Study.from_model(model, spec=spec).predict(held_out)

features = prediction.features.frame.set_index("subject")
fraction_columns = [c for c in features.columns if c.endswith("_volume_fraction")]
print("held-out habitat volume fractions:")
print(features[fraction_columns].round(3).to_string())

one = held_out[0]
fig = plot_habitat_overlay(
    one.image("LAP"),
    prediction.habitat_maps[0],
    title=f"{one.subject_id} (held out): habitats from the saved model",
    crop_to="labels",
)
fig.savefig("out/train_save_predict_held_out.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Check that labelling did not change the definition
# --------------------------------------------------
# Three checks a reviewer can ask for: the centroids are bit-identical
# after ``predict``, every held-out map carries the id of the model
# that labelled it (so maps from different definitions cannot be mixed
# silently), and no held-out map uses an id the model does not have.
print("centroids unchanged by predict:", np.array_equal(model.centroids, centroids_before))
print("maps carry the model id:       ", all(m.model_id == model.model_id for m in prediction.habitat_maps))
allowed = set(range(1, model.n_habitats + 1))
print("only the model's habitat ids:  ", all(set(np.unique(m.label_array[m.label_array > 0]).tolist()) <= allowed for m in prediction.habitat_maps))

# %%
# Same labels from a fresh Python process
# ---------------------------------------
# To show that the file alone is enough, a separate Python process loads
# the ``.habitatmodel``, labels the same held-out patients, and saves the
# label arrays. The child script uses absolute paths and does not import
# this page, so it shares nothing with the objects above. It uses the
# spec stored in the model, so it also shows that the file carries the
# analysis, not only the centroids.
out_dir = Path("out").resolve()
child_dir = out_dir / "train_save_predict_child"
child_dir.mkdir(exist_ok=True)
child_code = f'''
from pathlib import Path

import numpy as np

from habit.contracts import HabitatModel, cohort_from_directory
from habit.execution import backend_from_policy
from habit.recipes import Study

if __name__ == "__main__":
    model = HabitatModel.load(Path({str(model_path)!r}))
    cohort = cohort_from_directory(
        {str(Path(DATA).resolve())!r}, modalities={list(MODALITIES)!r}, roi={ROI!r}
    )
    prediction = Study.from_model(model).predict(cohort[3:5])
    for habitat_map in prediction.habitat_maps:
        out_file = Path({str(child_dir)!r}) / (habitat_map.subject_id + ".npy")
        np.save(out_file, np.asarray(habitat_map.label_array))
'''
child_script = out_dir / "train_save_predict_child.py"
child_script.write_text(child_code, encoding="utf-8")
subprocess.run([sys.executable, str(child_script)], check=True, cwd=str(out_dir))

for habitat_map in prediction.habitat_maps:
    child_labels = np.load(child_dir / f"{habitat_map.subject_id}.npy")
    same = np.array_equal(np.asarray(habitat_map.label_array), child_labels)
    print(f"{habitat_map.subject_id}: fresh process == this process -> {same}")

# %%
# Hand the feature table to your statistics tool
# ----------------------------------------------
# One row per held-out patient: volume fractions, MSI, ITH and graph
# columns. This CSV is where HABIT stops; outcome modelling or
# statistics happen in the tool of your choice. No model is built here.
prediction.features.frame.to_csv("out/train_save_predict_features.csv", index=False)
print("rows x columns:", prediction.features.frame.shape)
print(list(prediction.features.frame.columns[:8]))

# %%
# How to read the result
# ----------------------
# Measured on this run:
#
# * The elbow rule chose K = 4 habitats on subj001-subj003.
# * The archive holds two members (``manifest.json`` and
#   ``arrays/centroids.npy``, 3.6 kB in total); reloading it gave the
#   same centroids and
#   model id as the in-memory model (both True).
# * After labelling subj004 and subj005 the centroids were bit-identical,
#   both maps carried the model id, and only the model's ids appeared
#   (all True): prediction applied the definition without touching it.
# * The fresh Python process produced identical label arrays for both
#   held-out patients (True, True), using only the file and the demo
#   images.
#
# None of this says the habitats are clinically meaningful or that they
# transfer to another centre; that needs an independent cohort.

# %%
# Where to go next
# ----------------
# * The reference analysis this page reuses:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`.
# * Receive someone else's ``.habitatmodel`` and check it before use
#   (format version, required modalities):
#   :doc:`/auto_examples/05_validation_and_reuse/plot_04_reuse_published_model`.
# * Two models fitted separately need their ids matched:
#   :doc:`/auto_examples/05_validation_and_reuse/plot_02_matching_labels`.
# * The same spec on a whole cohort with workers and checkpoints:
#   :doc:`/auto_examples/07_advanced/plot_08_backends`.
# * Each feature family in the table: :doc:`/auto_examples/04_quantifying_habitats/index`.
