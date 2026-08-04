Spec, RunPolicy, and YAML isomorphism
=====================================

``habit.spec`` is how designs are declared, fingerprinted, saved, migrated, and
described — the same document shape as v1 YAML.

``Spec`` and ``HabitatSpec``
----------------------------

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

``RunPolicy``
-------------

.. code-block:: python

   from habit import RunPolicy, load_run_policy, save_run_policy

   policy = RunPolicy(
       workers=4,
       backend="process",              # "serial" | "process"
       on_subject_failure="continue",  # or "fail_fast"
       subject_timeout_sec=900.0,
   )
   save_run_policy(policy, "run_policy.yaml")
   policy2 = load_run_policy("run_policy.yaml")

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
