Spec, RunPolicy, and YAML isomorphism
=====================================

``habit.spec`` is how designs are declared, fingerprinted, saved, migrated, and
described — the same document shape as v1 YAML.

``Spec`` and ``HabitatSpec``
----------------------------

Keyword arguments may be written in **runtime pipeline order** (voxel features
→ optional voxel prep → supervoxels → optional cohort prep → fit → assign →
habitat features). That can differ from the dataclass field order, where
optional preprocessor chains appear later because they have defaults.

.. code-block:: python

   from habit import HabitatSpec, Spec, load_habitat_spec, save_habitat_spec

   spec = HabitatSpec(
       name="two_step",
       voxel_feature_extractor=Spec(
           name="raw",
           params={"modalities": ["T1", "T2"]},
       ),
       supervoxelizer=Spec(name="slic", params={"n_supervoxels": 50}),
       habitat_model_fitter=Spec(name="kmeans", params={"n_habitats": 4}),
       habitat_assigner=Spec(name="nearest_centroid"),
       habitat_features=(
           Spec(name="msi"),
           Spec(name="ith_score"),
           Spec(name="volume"),
       ),
       random_seed=42,
   )

   print(spec.fingerprint())
   print(spec.describe_methods(style="radiology"))
   print(spec.describe_methods(style="nature"))

   save_habitat_spec(spec, "habitat_spec.yaml")
   restored = load_habitat_spec("habitat_spec.yaml")
   payload = spec.to_dict()
   again = HabitatSpec.from_dict(payload)

Direct (no-supervoxel) design — set ``supervoxelizer=None``::

   direct = HabitatSpec(
       name="direct",
       voxel_feature_extractor=Spec("raw", {"modalities": ["T1"]}),
       supervoxelizer=None,
       habitat_model_fitter=Spec("kmeans", {"n_habitats": 3}),
       habitat_assigner=Spec("nearest_centroid"),
       habitat_features=(Spec("volume"),),
   )

Feature trees and the expression form
-------------------------------------

Extraction stages (``voxel_feature_extractor``,
``supervoxel_feature_extractor``, ``habitat_features``) accept a **tree** of
nodes: leaves carry ``modality=`` / ``modalities=`` parameters, and combiner
nodes nest their children under ``params["children"]`` as plain
``{"name", "params"}`` payloads. Any component entry may be written in two
**fingerprint-identical** spellings — the structured mapping above, or the
strict expression string parsed by :func:`~habit.spec.parse_feature_expression`
(:func:`~habit.spec.coerce_spec` routes a string entry to the parser and a
mapping entry to ``Spec.from_dict``):

.. code-block:: python

   from habit import HabitatSpec, parse_feature_expression

   expr = parse_feature_expression(
       'concat(raw("T1"), local_entropy("T2", kernel_size=3))'
   )
   spec = HabitatSpec(
       name="tree",
       voxel_feature_extractor=expr,
       # ... same remaining fields as above ...
   )

   # YAML dual form — a string entry is parsed the same way:
   again = HabitatSpec.from_dict(
       {"name": "tree", "voxel_feature_extractor":
        'concat(raw("T1"), local_entropy("T2", kernel_size=3))', ...}
   )
   assert again.fingerprint() == spec.fingerprint()

Expression grammar is deliberately strict: modality names are **quoted
strings**, parameters are explicit ``key=value`` literals, children are
nested calls (a quoted string among children becomes an implicit ``raw``
leaf). Bare v0.1-style identifiers are rejected with an explicit error
rather than guessed — the legacy YAML adapter keeps its permissive parser
for unquoted v0.1 expressions and only routes quoted expressions here, so
old configs translate byte-identically while new configs get the tree.

``RunPolicy``
-------------

:class:`~habit.spec.RunPolicy` is the declarative snapshot of every
scheduling concern. Field names match backend keyword arguments so the
YAML ``policy:`` block and the Python form stay one-to-one.

.. code-block:: python

   from habit import RunPolicy, load_run_policy, save_run_policy

   policy = RunPolicy(
       workers=4,
       backend="process",              # "serial" | "process"
       on_subject_failure="continue",  # or "fail_fast"
       subject_timeout_sec=900.0,
       parallel_mode="persistent",     # library default
       auto_retry_rounds=2,
   )
   save_run_policy(policy, "run_policy.yaml")
   policy2 = load_run_policy("run_policy.yaml")

``on_subject_failure="continue"`` isolates errors inside the execution
backend. :meth:`~habit.contracts.Cohort.map` still raises
``ProcessingError`` by default; pass ``raise_on_failure=False`` (recipes /
CLI) to proceed with successes — see :doc:`execution` and
:doc:`../examples/fault_tolerance`.

Full field set (defaults from ``habit/spec/policy.py``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Field
     - Default
     - Role
   * - ``workers``
     - ``1``
     - Parallel worker processes; with a positive timeout, ``1`` still uses ProcessPool
   * - ``backend``
     - ``"serial"``
     - ``"serial"`` or ``"process"`` (timeout also forces process at the CLI gate)
   * - ``subject_timeout_sec``
     - ``900.0``
     - Per-subject wall-clock seconds; ``None`` disables (ProcessPool when armed)
   * - ``subject_spawn_timeout_sec``
     - ``120.0``
     - Spawn-startup seconds; ``None`` disables (ProcessPool / isolated)
   * - ``graceful_shutdown_sec``
     - ``15.0``
     - Seconds between ``terminate()`` and ``kill()`` on timeout
   * - ``on_subject_failure``
     - ``"continue"``
     - ``"continue"`` or ``"fail_fast"`` (Serial + ProcessPool)
   * - ``oom_backoff``
     - ``True``
     - Reduce workers after fatal ``MemoryError`` (ProcessPool only)
   * - ``oom_reduce_workers_by``
     - ``1``
     - Workers subtracted per OOM step
   * - ``cap_workers_to_gpu_pool``
     - ``False``
     - Clamp workers to the usable GPU pool
   * - ``resume``
     - ``True``
     - Reuse checkpointed subject results when a store is attached
   * - ``checkpoint_dir``
     - ``None``
     - Checkpoint root; resolved by CLI/recipe (not applied by ``from_policy``)
   * - ``parallel_mode``
     - ``"persistent"``
     - ``"persistent"`` or ``"isolated"`` (ProcessPool only)
   * - ``auto_retry_rounds``
     - ``2``
     - In-run re-dispatch rounds for failed subjects; ``0`` disables
   * - ``retry_failed_subjects``
     - ``False``
     - Re-queue checkpointed failures on the next resumed run
   * - ``force_rerun_subjects``
     - ``()``
     - Subject IDs forced to recompute
   * - ``clear_checkpoint_on_success``
     - ``False``
     - Remove the checkpoint directory after a clean run
   * - ``strict_checkpoint_hash``
     - ``False``
     - Raise :class:`~habit.exceptions.CompatibilityError` on incompatible
       checkpoint fingerprint / legacy v0.1 layout (v0.1 parity)
   * - ``persistent_worker_max_consecutive_failures``
     - ``1``
     - Restart a persistent worker after this many consecutive fatal failures
   * - ``persistent_worker_recycle_after_tasks``
     - ``0``
     - Restart a persistent worker after this many successes (``0`` disables)

CLI / ``run_from_yaml`` select ProcessPoolBackend when
``backend == "process"`` **or** ``subject_timeout_sec`` is positive
(even for ``workers == 1``). Details: :doc:`execution`.

v0.1 YAML top-level keys vs ``RunPolicy``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **v0.1** habitat document keeps parallel knobs at the YAML top level
(``processes``, ``individual_subject_*``, …).
:class:`~habit.spec.legacy.LegacyConfigAdapter` renames them into the v1
``policy:`` section. A native **v1** document writes the right-hand names
under ``policy:`` directly (see ``config/habitat/config_habitat_two_step_v1.yaml``).
Habitat field reference: :doc:`../configuration/habitat`.

.. list-table::
   :header-rows: 1
   :widths: 38 30 16 16

   * - v0.1 top-level key
     - ``RunPolicy`` / v1 ``policy:`` field
     - Schema default
     - ``RunPolicy`` default
   * - ``processes``
     - ``workers``
     - ``2``
     - ``1``
   * - *(implied by* ``processes > 1`` *)*
     - ``backend``
     - —
     - ``"serial"`` (set to ``"process"`` when translated ``workers > 1``)
   * - ``individual_subject_timeout_sec``
     - ``subject_timeout_sec``
     - ``900.0``
     - ``900.0``
   * - ``individual_subject_spawn_timeout_sec``
     - ``subject_spawn_timeout_sec``
     - ``120.0``
     - ``120.0``
   * - ``individual_subject_graceful_shutdown_sec``
     - ``graceful_shutdown_sec``
     - ``15.0``
     - ``15.0``
   * - ``on_subject_failure``
     - ``on_subject_failure``
     - ``"continue"``
     - ``"continue"``
   * - ``oom_backoff``
     - ``oom_backoff``
     - ``True``
     - ``True``
   * - ``oom_reduce_workers_by``
     - ``oom_reduce_workers_by``
     - ``1``
     - ``1``
   * - ``cap_processes_to_gpu_pool``
     - ``cap_workers_to_gpu_pool``
     - ``False``
     - ``False``
   * - ``resume``
     - ``resume``
     - ``True``
     - ``True``
   * - ``checkpoint_dir``
     - ``checkpoint_dir``
     - ``None``
     - ``None``
   * - ``individual_subject_parallel_mode``
     - ``parallel_mode``
     - ``"persistent"``
     - ``"persistent"``
   * - ``individual_subject_auto_retry_rounds``
     - ``auto_retry_rounds``
     - ``2``
     - ``2``
   * - ``retry_failed_subjects``
     - ``retry_failed_subjects``
     - ``False``
     - ``False``
   * - ``force_rerun_subjects``
     - ``force_rerun_subjects``
     - ``[]``
     - ``()``
   * - ``clear_checkpoint_on_success``
     - ``clear_checkpoint_on_success``
     - ``False``
     - ``False``
   * - ``strict_checkpoint_hash``
     - ``strict_checkpoint_hash``
     - ``False``
     - ``False``
   * - ``persistent_worker_max_consecutive_failures``
     - ``persistent_worker_max_consecutive_failures``
     - ``1``
     - ``1``
   * - ``persistent_worker_recycle_after_tasks``
     - ``persistent_worker_recycle_after_tasks``
     - ``0``
     - ``0``

Note the default gap on ``processes`` / ``workers``: a bare v0.1 habitat
YAML defaults to ``processes: 2`` (process backend after translation), while
a bare ``RunPolicy()`` defaults to ``workers=1``, ``backend="serial"``.
CLI / ``run_from_yaml`` still select ProcessPool when the default
``subject_timeout_sec=900`` is armed.

Detect, validate, migrate YAML
------------------------------

.. code-block:: python

   from pathlib import Path
   import yaml

   from habit import (
       LegacyConfigAdapter,
       detect_yaml_version,
       migrate_yaml,
       validate_v1_document,
   )

   payload = yaml.safe_load(
       Path("config/habitat/config_habitat_two_step.yaml").read_text(
           encoding="utf-8"
       )
   )
   version = detect_yaml_version(payload)  # "v0" | "v1"

   # Structural validation of a v1 document
   # validate_v1_document(v1_payload, workflow="habitat")

   # Migrate v0 -> v1 (dry-run)
   report = migrate_yaml(
       "config/habitat/config_habitat_two_step.yaml",
       dry_run=True,
       workflow="habitat",
   )
   print(report.diff)
   print(report.document)

   # Lower-level translation
   translation = LegacyConfigAdapter().translate(payload, "habitat")
   # translation.document["spec"] / translation.document["policy"]

Run the translated spec with a recipe
-------------------------------------

v0.1 YAML selects the habitat design via
``habitat_segmentation.clustering_mode``. After translation, dispatch to the
matching L4 recipe (same table the CLI uses):

.. list-table::
   :header-rows: 1
   :widths: 30 40

   * - ``clustering_mode`` (YAML)
     - Recipe function
   * - ``two_step``
     - :func:`~habit.recipes.two_step`
   * - ``one_step``
     - :func:`~habit.recipes.one_step`
   * - ``direct_pooling``
     - :func:`~habit.recipes.direct_pooling`

Pattern: load the YAML, translate with :class:`~habit.spec.legacy.LegacyConfigAdapter`,
build a :class:`~habit.spec.specs.HabitatSpec`, then call the recipe named by
``clustering_mode`` (see :doc:`python_api`):

.. code-block:: python

   from pathlib import Path

   import yaml
   from habit import LegacyConfigAdapter, make_synthetic_cohort
   from habit.spec import HabitatSpec
   import habit.recipes as recipes

   payload = yaml.safe_load(
       Path("config/habitat/config_habitat_two_step.yaml").read_text(
           encoding="utf-8"
       )
   )
   translation = LegacyConfigAdapter().translate(payload, "habitat")
   spec = HabitatSpec.from_dict(translation.document["spec"])
   mode = payload["habitat_segmentation"]["clustering_mode"]

   _RECIPE_BY_MODE = {
       "two_step": recipes.two_step,
       "one_step": recipes.one_step,
       "direct_pooling": recipes.direct_pooling,
   }
   cohort = make_synthetic_cohort(n_subjects=4, rng=42)
   # cohort = DirectoryDataSource(...).load()  # real data on disk

   result = _RECIPE_BY_MODE[mode](cohort, spec)
   result.save("out/study")

Workflow aliases accepted by migrate / validate / adapter:

``preprocess``, ``habitat``, ``extract``, ``radiomics``, ``model``, ``cv``,
``compare``, ``icc``, ``retest``, ``sort-dicom``.

CLI
---

* ``habit check-config -c PATH`` — auto-detects v0/v1
* ``habit migrate-config -c PATH`` — writes a v1 document

See :doc:`../reference/cli`.
