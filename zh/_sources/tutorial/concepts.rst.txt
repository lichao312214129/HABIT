Concepts
========

**Background.** Habitat analysis has its own small vocabulary. The
Guide pages use these words without re-defining them at length, so this
page collects them in one place.

**Purpose.** After reading it you should be able to read any stage list
(:class:`~habit.spec.HabitatSpec`) and say what each stage does, which
statistics are learned from which patients, and what a saved model
contains.

**When to use.** Read it once before :doc:`/auto_examples/00_introductory_tutorials/index`,
and come back when a Guide page links a term here.

Images and regions
------------------

**Voxel.** One 3-D pixel of an image. HABIT only analyses voxels inside
the ROI.

**ROI / mask.** The region of interest, usually the whole tumour, stored
as an integer mask on the same grid as the images (0 = background). In
the demo the mask is keyed ``LAP``, the series it was drawn on.

**Modality / series.** One input image of a subject, for example a DCE
phase (``pre_contrast``, ``LAP``, ``PVP``, ``delay_3min``). All series of a
subject must share one voxel grid with the mask; that is image
preprocessing (:doc:`/how_to/preprocess`), which happens before habitat
analysis.

Features, supervoxels, habitats
-------------------------------

**Voxel feature.** The numbers that describe one voxel: its intensity in
each series (``raw``), or a local texture value computed around it
(``voxel_radiomics``). The ``extract`` stage produces one row per ROI
voxel and one column per feature.

**Supervoxel.** A small patch of neighbouring voxels with similar
features, found inside one subject by the ``partition`` stage (k-means or
SLIC). Its feature vector is the mean of its voxels. Clustering tens of
supervoxels per subject is faster and less noisy than clustering every
voxel.

**Habitat.** A group of voxels (or supervoxels) whose features are alike.
The ``fit`` stage learns the habitat definition (for k-means: the number
of habitats and one centroid per habitat); ``assign`` gives every voxel
the id of its nearest centroid. The result is a habitat map: every ROI
voxel painted with a habitat id 1, 2, 3, ...

**Habitat features.** Numbers computed from a habitat map, one row per
patient: volume fractions, MSI (how habitats touch each other), ITH
score (how fragmented the tumour is), graph metrics, radiomics per
habitat. Definitions: :doc:`/reference/features/index`.

Subject-level and cohort-level
------------------------------

Every step is either **subject-level** (it sees one patient at a time)
or **cohort-level** (it sees the pooled rows of all training patients).
The ``pool`` stage is the boundary.

* **Subject-level** stages run before ``pool``: ``extract``, a
  ``preprocess`` stage placed before ``pool``, and ``partition``. They use
  only that patient's own data, store no state, and cannot leak
  information from one patient into another. A new patient goes through
  them exactly like a training patient.
* **Cohort-level** stages run after ``pool``: a ``preprocess`` stage placed
  after ``pool``, and ``fit``. They are learned once on the training
  patients. Their state (for example bin edges, and the centroids) is
  saved inside the model and reused **unchanged** on new patients; it is
  never refitted on a new patient.

The same component name can sit on either side. Its position in the
stage list decides which one it is.

Why preprocessing matters
-------------------------

MR intensities are arbitrary units: two scanners, or two sessions on one
scanner, can give the same tissue very different grey values. Clustering
raw intensities across patients then groups patients by overall
brightness instead of by the pattern inside each tumour.

The demo shows it. Fit on two patients with raw intensities, and a third
patient whose scan is globally darker (ROI median 545 in the arterial
phase versus 864 and 976 for the training patients) gets about 85 % of
its voxels in the darkest habitat -- one big block. Cohort-level
preprocessing alone does not fix this (the largest habitat still covers
52-67 %). Subject-level normalization does: winsorize each feature to
the patient's own 5th-95th percentile, then min-max it to 0..1, and the
new patient shows all four habitats (fractions 0.09 to 0.40). That is
the definition used in :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`
and the quickstart.

The trade-off: per-subject min-max removes the absolute whole-tumour
enhancement level. Habitats then describe the pattern inside each
tumour, not how bright the tumour is overall. If absolute level matters
for your question, report it as a separate whole-ROI feature.

Three habitat designs
---------------------

The stage list decides the design (:doc:`/auto_examples/00_introductory_tutorials/index`):

* **Two-step** -- ``partition`` then ``pool`` then ``fit``. Supervoxels per
  patient, one model for the cohort. Habitat ids mean the same thing in
  every patient. This is the default.
* **One-step** -- no ``pool``: each patient is clustered on its own.
  Habitat ids are not comparable across patients; use label-free
  summaries (ITH score, number of habitats) or match labels first
  (:doc:`/auto_examples/05_validation_and_reuse/plot_02_match_same_subject`).
* **Direct pooling** -- ``pool`` then ``fit`` with no ``partition``: every
  voxel of every patient is clustered together. Shared ids, but the
  matrix grows with every voxel.

Declaring an analysis
---------------------

**Spec.** A registered component name plus its parameters, for example
``Spec("kmeans", {"n_supervoxels": 30})``. It round-trips to YAML and has
a fingerprint, so a result can say exactly what produced it.

**Stage.** One step of the analysis: ``Stage("partition", Spec(...))``.
The first argument is a label you choose; HABIT infers the role
(extract, preprocess, partition, pool, fit, assign, quantify) from the
component and its position.

**HabitatSpec.** The ordered list of stages plus ``random_seed``: the
whole study definition. :class:`~habit.recipes.Study` runs it
(``fit_predict`` on a training cohort, ``predict`` on new patients). A
YAML file's ``spec:`` block is the same object.

**.habitatmodel.** The saved habitat definition: centroids, the fitted
cohort-level preprocessing state, the full stage list, the defining
cohort's fingerprint and software versions.
:meth:`HabitatModel.load() <habit.contracts.HabitatModel.load>` reads it
back and labels new patients without refitting
(:doc:`/auto_examples/05_validation_and_reuse/plot_01_train_save_predict`).

**Methods paragraph.** ``result.manifest.describe_methods()`` drafts a
manuscript methods paragraph from what actually ran
(:doc:`/auto_examples/05_validation_and_reuse/plot_01_train_save_predict`).

Running a cohort
----------------

**Backend.** What schedules the subjects: serial (one after another in
this process) or a pool of worker processes. Changing the backend
changes speed, never the habitat labels.

**RunPolicy.** The settings of a run: backend, workers, what to do when
one subject fails (``on_subject_failure``), per-subject timeout.

**Checkpoint.** Finished subject results written to disk
(:class:`~habit.execution.CheckpointStore`), so an interrupted run
resumes without recomputing them.

Examples: :doc:`/auto_examples/07_advanced/plot_08_backends` and
:doc:`/auto_examples/07_advanced/plot_08_backends`.

Next
----

:doc:`/auto_quickstart/plot_quickstart_python`, then
:doc:`/auto_examples/00_introductory_tutorials/index`.
