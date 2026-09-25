"""
Two-step habitats from atomic components
========================================

**Background.** :doc:`/auto_examples/01_complete/plot_01_two_step_spec`
declares a whole habitat study in one :class:`~habit.spec.HabitatSpec`
and runs it with one ``Study(spec).fit_predict(...)`` call. Behind that
call sit a handful of ordinary Python objects: a voxel feature
extractor, two preprocessing chains, a supervoxelizer, a habitat model
fitter, an assigner and four habitat feature extractors. Each of them
is public and can be called on one subject, in memory, without a spec,
a YAML file or an output folder.

**Purpose.** You will rebuild the approved two-step analysis of
:doc:`/auto_examples/01_complete/plot_01_two_step_spec` (two training
patients, four DCE phases, ROI ``LAP``, seed 0) from those components,
as a plain per-subject loop. Then ``Study`` runs the same spec, and the
page checks that both paths give the same habitat labels voxel for
voxel, the same model id and the same feature table. Finally the
atomic path labels a third, new patient, and the atomic model is
handed to ``Study.from_model`` to label the same patient again.

**When to use.** When you want to see what ``Study`` does inside, or
when you need a single step in your own pipeline -- a MONAI transform
chain, a notebook, a service -- and do not want HABIT's orchestration
around it. For a normal study, ``Study`` is shorter and records the
run manifest for you.

**Analysis definition.** The approved two-step definition, unchanged:
raw intensities of four phases, subject-level winsorize + min-max, 30
k-means supervoxels, cohort-level 10-bin uniform binning, k-means with
2..10 habitats chosen by the elbow rule, nearest-centroid assignment,
then volume, MSI, ITH and graph features. Nothing is varied; only the
way the steps are called changes.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **voxel field** -- the table of voxel features of one subject (one
  row per ROI voxel, one column per feature), plus where each voxel
  sits in the image.
* **subject-level chain** -- preprocessing that uses each patient's
  own statistics; it has no fitted state, so it is simply called.
* **supervoxelization** -- the supervoxel label map of one subject and
  one feature row per supervoxel.
* **cohort-level chain** -- preprocessing fitted once on the pooled
  training rows; its fitted state must be stored with the model.
* **fitter / model / assigner** -- the fitter learns centroids from
  the pooled rows and returns a ``HabitatModel``; the model's assigner
  paints habitat ids onto one subject.
* **habitat feature extractor** -- turns one subject's habitat map
  into one row of the feature table.

Two things ``Study`` does for you are also done here, just less
visibly. First, it aligns each ROI mask onto the image grid; the raw
voxel feature extractor does that itself (the demo masks have a
different origin and direction from their images), so calling the
extractor directly gets the same aligned voxels. Second, ``Study``
seeds every stochastic component with ``spec.random_seed``; below the
same seed is passed to each component with ``set_random_state(0)``,
which is why the two paths can agree bit for bit.
"""

# %%
# Load the cohort
# ---------------
# Same data, same split as the Spec page: two training patients
# (subj001, subj002) and one new patient (subj003).
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.feature_preprocessing import (
    Binning,
    CohortPreprocessingChain,
    MinMaxScaling,
    SubjectPreprocessingChain,
    Winsorizing,
)
from habit.habitat_features import (
    GraphHabitatFeatures,
    HabitatVolumeFeatures,
    IthHabitatFeatures,
    MsiHabitatFeatures,
)
from habit.habitat_model import KMeansHabitatModelFitter
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.supervoxel import KMeansSupervoxelizer
from habit.viz import plot_habitat_overlay, plot_partition_triptych
from habit.voxel_features import RawVoxelFeatures

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train, new_patient = cohort[:2], cohort[2:3]
# The one seed used for every random step. Study uses spec.random_seed.
SEED = 0
Path("out").mkdir(exist_ok=True)

# %%
# Build the components
# --------------------
# Each object below is the Python twin of one stage in the spec of
# :doc:`/auto_examples/01_complete/plot_01_two_step_spec`. Nothing is
# computed yet: the objects only hold their settings.

# extract: one intensity column per DCE phase, voxels inside the ROI.
# The extractor also resamples the ROI mask onto the image grid when
# their geometry differs.
extractor = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)

# preprocess + preprocess2 (subject-level): clip each column to that
# patient's own 5th..95th percentile, then rescale it to 0..1 with that
# patient's own min and max. The chain has no fitted state: calling it
# on a patient only ever uses that patient's numbers. HABIT puts a
# mean imputation first on its own, so missing values never reach the
# scalers (it shows up in the chain's spec).
subject_chain = SubjectPreprocessingChain([Winsorizing(winsor_limits=(0.05, 0.05)), MinMaxScaling()])

# partition: k-means with 30 supervoxels inside each tumour. k-means
# starts from random centres, so the seed is fixed; without it a rerun
# could draw different supervoxel borders.
supervoxelizer = KMeansSupervoxelizer(n_supervoxels=30)
supervoxelizer.set_random_state(SEED)

# fit: one cohort k-means that tries 2..10 habitats and keeps the elbow.
# n_init=10 restarts every candidate count from 10 random starts. It is
# seeded for the same reason as the supervoxelizer.
fitter = KMeansHabitatModelFitter(min_habitats=2, max_habitats=10, validation="elbow", n_init=10)
fitter.set_random_state(SEED)

# quantify: four habitat feature families, each giving one row per subject.
feature_extractors = [
    HabitatVolumeFeatures(),
    MsiHabitatFeatures(),
    IthHabitatFeatures(),
    GraphHabitatFeatures(include_extended_metrics=False),
]
print(subject_chain.spec)

# %%
# Step 1: per-subject features and supervoxels
# --------------------------------------------
# An explicit loop over subjects. Every call takes ONE subject (or one
# subject's result) and returns an in-memory object, so any line of
# this loop can be lifted into another pipeline on its own.
#
# ``with_feature_frame`` swaps the feature table of the voxel field for
# the preprocessed one and records which step produced it (the
# ``produced_by`` label and the chain's spec fingerprint go into the
# provenance of everything computed afterwards).
units = []
for subject in train:
    # Voxel field: one row per ROI voxel, one column per phase.
    field = extractor(subject)
    # Subject-level normalization with this patient's own statistics.
    field = field.with_feature_frame(
        subject_chain(field.feature_frame()),
        produced_by="feature_preprocessing.subject.voxel",
        spec_fingerprint=subject_chain.spec.fingerprint(),
    )
    # 30 supervoxels inside this tumour; one mean feature row per supervoxel.
    units.append(supervoxelizer(field))
    print(subject.subject_id, "voxels:", field.feature_frame().shape[0], "supervoxels:", units[-1].feature_frame().shape[0])

# %%
# Step 2: pool and fit the cohort-level chain
# -------------------------------------------
# ``pool`` is just a concatenation of every training subject's
# supervoxel rows. The cohort-level chain (10 equal-width bins per
# feature) learns its bin edges from these pooled TRAINING rows only.
# If the new patient's rows were included, information from the patient
# we want to label would leak into the definition, and the definition
# would change every time a patient is added.
pooled = pd.concat([one.feature_frame() for one in units], ignore_index=True)
cohort_chain = CohortPreprocessingChain([Binning(n_bins=10, bin_strategy="uniform")])
cohort_chain.fit(pooled)
print("pooled training rows:", pooled.shape)

# The fitted chain is then APPLIED (not refitted) to each subject's rows.
binned_units = [
    one.with_feature_frame(
        cohort_chain.transform(one.feature_frame()),
        produced_by="feature_preprocessing.cohort",
        spec_fingerprint=cohort_chain.spec.fingerprint(),
    )
    for one in units
]

# %%
# Step 3: fit the habitat model and assign labels
# -----------------------------------------------
# The fitter clusters the binned supervoxel rows once for the whole
# training cohort and returns a ``HabitatModel`` (the centroids).
#
# The centroids live in the BINNED feature space. A new patient can only
# be compared with them after the same bin edges are applied, so the
# fitted chain's state is attached to the model with
# ``with_cohort_preprocessing``. That is what makes a saved
# ``.habitatmodel`` self-contained: whoever loads it gets the centroids
# and the bin edges together.
model = fitter.fit(binned_units, cohort=train)
model = model.with_cohort_preprocessing(cohort_chain.state, cohort_chain.spec.to_dict())
print(model.summary())

# The assigner gives every supervoxel the id of its nearest centroid,
# then writes that id onto the supervoxel's voxels.
assigner = model.assigner("nearest_centroid")
atomic_maps = [assigner(one) for one in binned_units]

subject = train[0]
fig = plot_partition_triptych(subject.image("LAP"), units[0], atomic_maps[0], axis=0)
fig.savefig("out/atomic_triptych.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Step 4: the feature table
# -------------------------
# Every habitat feature extractor is called as ``extractor(subject,
# habitat_map)`` and returns a one-row table. ``join`` glues the four
# families side by side; ``concat`` stacks the subjects.
rows = []
for one, one_map in zip(train, atomic_maps):
    table = feature_extractors[0](one, one_map)
    for feature_extractor in feature_extractors[1:]:
        table = table.join(feature_extractor(one, one_map))
    rows.append(table.frame)
atomic_table = pd.concat(rows, ignore_index=True)
print(atomic_table.shape[1], "columns")
print(atomic_table[["subject", "habitat_1_volume_fraction", "habitat_2_volume_fraction", "ith_score"]].round(3).to_string())

# %%
# Check against Study
# -------------------
# The approved spec, run with ``Study``. Same data, same seed, so the
# labels must match voxel for voxel and the model id (a digest of the
# fitter settings and the training subjects) must be the same. Each
# check prints True or False.
spec = HabitatSpec(
    name="full_pipeline_two_step",
    stages=(
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.05, 0.05]})),
        Stage("preprocess2", Spec("minmax")),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
        Stage("pool", Spec("pool")),
        Stage("preprocess_cohort", Spec("binning", {"n_bins": 10, "bin_strategy": "uniform"})),
        Stage("fit", Spec("kmeans", {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10})),
        Stage("assign", Spec("nearest_centroid")),
        Stage("volume", Spec("volume")),
        Stage("msi", Spec("msi")),
        Stage("ith", Spec("ith_score")),
        Stage("graph", Spec("graph", {"include_extended_metrics": False})),
    ),
    random_seed=SEED,
)
study = Study(spec)
study_result = study.fit_predict(train)
print("atomic model:", model.model_id, "| Study model:", study_result.habitat_model.model_id)
print("same model id:", model.model_id == study_result.habitat_model.model_id)
print("same number of habitats:", model.n_habitats == study_result.habitat_model.n_habitats)
for atomic_map, study_map in zip(atomic_maps, study_result.habitat_maps):
    same = np.array_equal(atomic_map.label_array, study_map.label_array)
    print(f"{atomic_map.subject_id}: labels identical = {same}")

study_table = study_result.features.frame
same_columns = list(atomic_table.columns) == list(study_table.columns)
numeric = atomic_table.select_dtypes("number").columns
same_values = np.allclose(
    atomic_table[numeric].to_numpy(dtype=float),
    study_table[numeric].to_numpy(dtype=float),
    equal_nan=True,
)
print(f"feature table: same columns = {same_columns}, values allclose = {same_values}")

# %%
# Label a new patient with the atomic path
# ----------------------------------------
# A new patient goes through the same per-subject steps (its own
# subject-level normalization, its own 30 supervoxels). Then the
# TRAINING bin edges and the TRAINING centroids are applied; nothing is
# refitted. The bin edges are read back from the model here, exactly as
# someone would after ``HabitatModel.load(...)``.
restored_chain = CohortPreprocessingChain.from_state(model.preprocessing_state["cohort_feature_preprocessor"])
new_subject = new_patient[0]
field = extractor(new_subject)
field = field.with_feature_frame(
    subject_chain(field.feature_frame()),
    produced_by="feature_preprocessing.subject.voxel",
    spec_fingerprint=subject_chain.spec.fingerprint(),
)
new_units = supervoxelizer(field)
new_units = new_units.with_feature_frame(
    restored_chain.transform(new_units.feature_frame()),
    produced_by="feature_preprocessing.cohort",
    spec_fingerprint=restored_chain.spec.fingerprint(),
)
new_map = assigner(new_units)

# Same patient through the fitted Study, for comparison.
study_new = study.predict(new_patient)
print(
    f"{new_subject.subject_id}: atomic == Study.predict: "
    f"{np.array_equal(new_map.label_array, study_new.habitat_maps[0].label_array)}"
)

fig = plot_habitat_overlay(
    new_subject.image("LAP"),
    new_map,
    title=f"{new_subject.subject_id}: habitats from the atomic path",
    crop_to="labels",
)
fig.savefig("out/atomic_new_patient.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Hand the atomic model to Study
# ------------------------------
# The model built above can also drive ``Study.predict``. A model fitted
# directly with ``fitter.fit(...)`` holds the centroids and the cohort
# bin edges, but not the upstream stages (which voxel features, which
# subject-level scaling, how many supervoxels). ``spec=spec`` supplies
# them, so ``Study`` puts the new patient through the same stages before
# the stored bin edges and centroids are applied.
from_model = Study.from_model(model, spec=spec).predict(new_patient)
print(
    f"{new_subject.subject_id}: atomic == Study.from_model(atomic model): "
    f"{np.array_equal(new_map.label_array, from_model.habitat_maps[0].label_array)}"
)
inside = np.asarray(new_map.label_array)
inside = inside[inside > 0]
ids, counts = np.unique(inside, return_counts=True)
for hid, count in zip(ids, counts):
    print(f"{new_subject.subject_id} habitat {int(hid)}: {count / inside.size:.3f} of ROI voxels")

# %%
# How to read the result
# ----------------------
# When this page was built, the atomic path produced:
#
# * 30 supervoxels per training patient (34694 ROI voxels in subj001,
#   9858 in subj002), 60 pooled rows with 4 feature columns.
# * K = 4 habitats chosen by the elbow rule, and the same model id as
#   ``Study`` (``same model id: True``).
# * Every "identical" / "allclose" check above printed ``True``: the
#   label maps of both training patients, the 453-column feature table,
#   and the new patient's map from the atomic path,
#   ``Study.predict`` and ``Study.from_model(model, spec=spec)``.
# * The new patient (subj003) uses all four habitats, with ROI fractions
#   0.191, 0.321, 0.396 and 0.092 for habitats 1 to 4.
#
# The checks show that the components reproduce ``Study`` exactly on
# this data and seed. They say nothing about whether K = 4 is a good
# habitat definition; two patients are a demo, not a cohort.

# %%
# Where to go next
# ----------------
# * The other two designs (inside each subject, pooled voxels) built
#   from components: :doc:`/auto_examples/01_complete/plot_05_atomic_other_designs`.
# * These components on serial and process-pool backends:
#   :doc:`/auto_examples/01_complete/plot_06_atomic_on_backends`.
# * Each component on its own page, with its options:
#   :doc:`/auto_examples/02b_stages/index` (supervoxels:
#   :doc:`/auto_examples/02b_stages/plot_12_supervoxels`, fitting:
#   :doc:`/auto_examples/02b_stages/plot_14_fit_model`, assigning:
#   :doc:`/auto_examples/02b_stages/plot_15_assign_labels`).
# * Every class used here: :doc:`/api/index`.
