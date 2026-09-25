"""
A Spec on execution backends
============================

**Background.** A :class:`~habit.spec.HabitatSpec` says *what* to
compute. *How* the subjects are scheduled is a separate choice: one
after another in this process, or spread over worker processes; stop at
the first broken subject or carry on; resume an interrupted run from
disk; stop a subject that hangs. None of these choices may change the
science, so the habitat labels must come out identical whichever
backend runs the spec.

**Purpose.** You will run one spec serially and with two worker
processes, check that both give the same habitat labels, rerun it from
a checkpoint store, add a broken subject and see it recorded while the
other subjects keep their labels, trigger a per-subject timeout, and
measure why the worker processes are not faster on this tiny cohort.

**When to use.** Any time the cohort is more than a handful of
patients, or one bad file or one hanging subject should not cost you
the whole run.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **RunPolicy** -- the scheduling settings: backend, number of
  workers, failure handling, timeouts, retries, resume.
* **backend** -- ``SerialBackend()`` runs subjects one after another in
  this process; ``backend_from_policy(RunPolicy(backend="process"))``
  builds a ``ProcessPoolBackend``.
* **spawn** -- how Windows starts a worker: a fresh Python interpreter
  that re-imports HABIT, which costs seconds per worker.
* **checkpoint** -- per-subject results written to disk, so a rerun
  loads them instead of recomputing.
* **subject outcome** -- the run manifest's per-subject record of
  success or failure (with the error message).

The analysis definition is the two-step definition of
:doc:`/auto_examples/01_complete/plot_01_two_step_spec`, unchanged:
``extract`` (raw intensities of the four DCE phases inside ROI
``LAP``), per-subject ``winsorize`` (5 % at each end) and ``minmax``,
``partition`` (k-means, 30 supervoxels), ``pool``, cohort-level
``binning`` (10 equal-width bins, stored in the model), ``fit``
(k-means, 2..10 habitats chosen by the elbow rule, ``n_init=10``),
``assign`` (nearest centroid), and the ``volume`` / ``msi`` /
``ith_score`` / ``graph`` habitat features, with seed 0. Only the
scheduling changes on this page.

.. note::

   The page runs every configuration several times, so it uses the
   three demo patients with the smallest lesions (subj002, subj003,
   subj004); subj001 and subj005 would dominate the run time. Three
   demo patients are a demo, not a cohort; the habitat count is picked
   by the elbow rule on these three, so it can differ from the
   two-patient fit of the two-step page.

On Windows a process pool must start under ``if __name__ == "__main__":``
(each worker re-imports this file), so every cell below is written that
way and can be copied as is. To send your own per-subject function to
worker processes, see
:doc:`/auto_examples/01_complete/plot_06_atomic_on_backends`.
"""

# %%
# Load the cohort and declare the study
# -------------------------------------
# The spec is the two-step spec, unchanged (elbow fit, seed 0). The
# elbow fit is deterministic for a fixed seed, so serial and process
# runs can be compared label for label.
# sphinx_gallery_thumbnail_number = 1
import dataclasses
import operator
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.exceptions import ProcessingError
from habit.execution import CheckpointStore, SerialBackend, backend_from_policy
from habit.recipes import Study
from habit.spec import HabitatSpec, RunPolicy, Spec, Stage
from habit.viz import plot_habitat_overlay

# Change DATA / MODALITIES / ROI to your preprocessed layout.
# The guard keeps a spawned worker from loading the cohort again.
if __name__ == "__main__":
    DATA = fetch_demo()
    MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
    ROI = "LAP"
    cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
    # subj002, subj003, subj004 (the smallest lesions); see the note at the top.
    cohort = cohort[1:4]
    spec = HabitatSpec(
        name="two_step_on_backends",
        stages=(
            Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
            # subject-level: each patient clipped and scaled with its own statistics.
            Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.05, 0.05]})),
            Stage("preprocess2", Spec("minmax")),
            Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
            Stage("pool", Spec("pool")),
            # cohort-level: bin edges learned on the pooled supervoxels, stored in the model.
            Stage("preprocess_cohort", Spec("binning", {"n_bins": 10, "bin_strategy": "uniform"})),
            Stage("fit", Spec("kmeans", {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10})),
            Stage("assign", Spec("nearest_centroid")),
            Stage("volume", Spec("volume")),
            Stage("msi", Spec("msi")),
            Stage("ith", Spec("ith_score")),
            Stage("graph", Spec("graph", {"include_extended_metrics": False})),
        ),
        random_seed=0,
    )
    print(f"Cohort: {list(cohort.subject_ids)}")

# %%
# Serial reference run
# --------------------
# One subject after another in this process. This is the reference the
# process run must reproduce.
if __name__ == "__main__":
    timings: dict[str, float] = {}
    start = time.perf_counter()
    serial = Study(spec).fit_predict(cohort, backend=SerialBackend())
    timings["serial"] = time.perf_counter() - start
    print(f"serial: {timings['serial']:.1f} s, K = {serial.habitat_model.n_habitats}")

# %%
# Worker processes with a checkpoint store
# ----------------------------------------
# ``RunPolicy`` collects the scheduling settings:
#
# * ``backend="process", workers=2`` -- two worker processes share the
#   subjects.
# * ``on_subject_failure="continue"`` -- a failing subject is recorded
#   and skipped instead of stopping the run.
# * ``subject_timeout_sec=600`` -- a worker busy with one subject for
#   more than 10 minutes is stopped and that subject counts as failed
#   (demonstrated further down).
# * ``auto_retry_rounds=0`` -- by default a failed subject is retried
#   twice in the same run (useful after an out-of-memory crash); the
#   failure on this page is a missing file, which a retry cannot fix.
#
# ``CheckpointStore`` keeps each finished subject on disk. The first
# call computes and stores; the second call with the same store loads
# the stored subjects instead of recomputing them. The store here is a
# fresh temporary folder so the first run really starts from nothing;
# in a study you would point it at a folder you keep.
if __name__ == "__main__":
    policy = RunPolicy(
        backend="process",
        workers=2,
        on_subject_failure="continue",
        subject_timeout_sec=600,
        auto_retry_rounds=0,
    )
    backend = backend_from_policy(policy)
    print(type(backend).__name__, "workers:", backend.workers)
    store = CheckpointStore(Path(tempfile.mkdtemp(prefix="habit_spec_backends_")))

    start = time.perf_counter()
    process = Study(spec).fit_predict(cohort, backend=backend, checkpoint=store)
    timings["process, first run"] = time.perf_counter() - start

    start = time.perf_counter()
    resumed = Study(spec).fit_predict(cohort, backend=backend, checkpoint=store)
    timings["process, rerun from checkpoints"] = time.perf_counter() - start

    for name, seconds in timings.items():
        print(f"{name}: {seconds:.1f} s")
    print("model ids:", serial.habitat_model.model_id, process.habitat_model.model_id, resumed.habitat_model.model_id)
    for left, right, again in zip(serial.habitat_maps, process.habitat_maps, resumed.habitat_maps):
        print(
            f"{left.subject_id}: process == serial: {np.array_equal(left.label_array, right.label_array)}, "
            f"rerun == serial: {np.array_equal(left.label_array, again.label_array)}"
        )

# %%
# The cohort feature table
# ------------------------
# One row per patient, from the process run. The same colour (habitat
# id) means the same habitat in every patient because the model was
# fitted once on the pooled supervoxels of all three.
if __name__ == "__main__":
    table = process.features.frame.set_index("subject")
    fractions = [c for c in table.columns if c.endswith("_volume_fraction")]
    print(table[fractions + ["ith_score"]].round(3).to_string())

    subject, habitat_map = cohort[0], process.habitat_maps[0]
    Path("out").mkdir(exist_ok=True)
    fig = plot_habitat_overlay(
        subject.image("LAP"),
        habitat_map,
        title=f"{subject.subject_id}: habitats (3-patient model, process run)",
        crop_to="labels",
    )
    fig.savefig("out/spec_backends_habitats.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# When one subject is broken
# --------------------------
# Below, a copy of one demo patient is added without its portal-venous
# series, so ``extract`` cannot find ``PVP`` for that copy. With
# ``on_subject_failure="continue"`` the run records the failure in
# ``manifest.subject_outcomes``, leaves the copy out of the habitat
# fit, and still returns maps for every other patient. With
# ``"fail_fast"`` it would stop at the first failure instead.
#
# Because the broken copy is left out, the habitat fit sees the same
# three patients as before, so their labels do not change. The same
# checkpoint store is passed, so the three good patients are loaded
# from it and only the broken copy is attempted.
if __name__ == "__main__":
    source = cohort[2]
    broken = dataclasses.replace(
        source,
        subject_id=f"{source.subject_id}_no_PVP",
        images={name: ref for name, ref in source.images.items() if name != "PVP"},
    )
    incomplete = Cohort(list(cohort) + [broken], name="one_incomplete")
    continued = Study(spec).fit_predict(incomplete, backend=backend, checkpoint=store)
    print("maps returned for:", [one.subject_id for one in continued.habitat_maps])
    for subject_id, outcome in continued.manifest.subject_outcomes.items():
        print(f"  {subject_id}: {outcome}")
    for left, right in zip(serial.habitat_maps, continued.habitat_maps):
        print(f"{left.subject_id}: unchanged by the failure: {np.array_equal(left.label_array, right.label_array)}")

# %%
# A per-subject timeout
# ---------------------
# ``subject_timeout_sec`` is enforced by the process backend: the parent
# measures how long a worker has been busy with one subject and, past
# the budget, terminates that worker, records a ``SubjectTimeoutError``
# for the subject and starts a fresh worker for the rest. The serial
# backend has no separate process to stop, so it has no timeout.
#
# To see it happen, the fitted model labels one patient with a budget
# of 0.1 s -- far too short for any real subject. In a study, pick a
# budget a few times longer than your slowest normal subject.
#
# With ``"continue"`` a timed-out subject is recorded like any other
# failure. Here it is the ONLY subject, and a run in which every
# subject failed has nothing to return, so the recipe raises
# ``ProcessingError`` naming each failed subject and its error.
if __name__ == "__main__":
    strict = backend_from_policy(
        RunPolicy(backend="process", workers=1, on_subject_failure="continue", subject_timeout_sec=0.1, auto_retry_rounds=0)
    )
    try:
        Study.from_model(serial.habitat_model, spec=spec).predict(cohort[2:3], backend=strict)
    except ProcessingError as error:
        print("stopped:", error)

# %%
# Why serial wins on this tiny cohort
# -----------------------------------
# To separate the process machinery from the analysis, the same process
# backend maps ``operator.attrgetter("subject_id")`` -- a function that
# does no image work -- over the same three subjects. That time is pure
# overhead: on Windows each worker is a new Python interpreter that
# imports HABIT and its dependencies (``spawn``), and every subject is
# pickled and sent to a worker, which then reads that subject's images
# from disk itself.
#
# A second cost is threading. In the serial run numpy / scikit-learn
# use every CPU core for k-means; each worker is capped to ONE
# BLAS/OpenMP thread (environment variable ``HABIT_WORKER_THREADS``,
# default 1) so that ``workers x threads`` cannot oversubscribe the
# machine. And the habitat fit itself is one cohort-level step that runs
# in this process whatever the backend.
#
# The bar chart is a cost demonstration on three subjects, not a claim
# that serial is always faster. Workers pay off when there are many
# more subjects than workers and each subject is expensive.
if __name__ == "__main__":
    start = time.perf_counter()
    ids = sorted(one.value for one in backend.map(operator.attrgetter("subject_id"), cohort))
    timings["process, no work (start-up only)"] = time.perf_counter() - start
    print("returned ids:", ids)
    for name, seconds in timings.items():
        print(f"{name}: {seconds:.1f} s")

    fig, ax = plt.subplots(figsize=(6.8, 2.6), constrained_layout=True)
    ax.barh(list(timings), list(timings.values()), color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel(f"wall-clock seconds ({len(cohort)} subjects)")
    ax.set_title("One HabitatSpec: serial vs 2 worker processes")
    fig.savefig("out/spec_backends_timing.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# How to read the result
# ----------------------
# In the reference run of this page (Windows, 2 workers):
#
# * RESULTS_PLACEHOLDER

# %%
# Where to go next
# ----------------
# * Your own per-subject function on the same backends:
#   :doc:`/auto_examples/01_complete/plot_06_atomic_on_backends`.
# * Train on some patients and label held-out ones with the saved
#   model: :doc:`/auto_examples/01_complete/plot_18_train_save_predict`.
# * The same analysis from individual components:
#   :doc:`/auto_examples/01_complete/plot_04_atomic_two_step`.
# * Every ``RunPolicy`` setting: :doc:`/tutorial/execution`.
