Quickstart: Python API
========================

Install first (:doc:`installation`). This page is the **same demo** as
:doc:`quickstart`, expressed as **pure Python** — construct a cohort,
:class:`~habit.HabitatSpec` / :class:`~habit.MLSpec`, and call
:mod:`habit.recipes` with explicit arguments (no YAML path, no
:func:`~habit.recipes.run_from_yaml`).

Parameters below mirror the bundled demo configs
(``config/habitat/config_habitat_two_step.yaml``,
``config/feature_extraction/config_extract_features_demo.yaml``,
``config/machine_learning/config_machine_learning_radiomics_minimal.yaml``)
so habitat label maps match the CLI voxel-wise when you use the same
``demo_data/``, seed, and execution policy.

YAML / CLI remain an alternate shell for the same recipes
(:doc:`quickstart`); they are not required here. For the recipe → atomic →
custom layering of habitat analysis, see
:doc:`../examples/habitat_analysis_overview`.

Work from a directory that already has ``demo_data/`` (and optionally
``config/`` if you also use the CLI). See :doc:`quickstart` steps 1–2::

   # Windows - Anaconda Prompt
   conda activate habit
   cd D:\my_habit_work          # your work_dir (has demo_data/)

1. Habitat analysis (two-step)
------------------------------

CLI twin (same scientific settings)::

   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

Save the block below as a ``.py`` file (for example ``run_two_step.py``)
and run ``python run_two_step.py``. Do **not** paste it at the top level
of a script without the ``__main__`` guard — the demo uses a process
pool (same as the CLI YAML ``processes: 2``).

.. include:: ../_includes/windows_multiprocessing.rst

.. code-block:: python

   from pathlib import Path

   from habit import HabitatSpec, RunPolicy, Spec, Stage, cohort_from_directory
   from habit.execution import backend_from_policy
   import habit.recipes as recipes


   def main() -> None:
       modalities = ("pre_contrast", "LAP", "PVP", "delay_3min")
       # Match the CLI YAML loader: ROI key = first modality in the concat list.
       cohort = cohort_from_directory(
           "demo_data/preprocessed",
           modalities=modalities,
           roi="pre_contrast",
       )

       # Same science as config/habitat/config_habitat_two_step.yaml,
       # declared as ordered stages (source of truth). Strategy is inferred:
       # partition + pool → two_step.
       spec = HabitatSpec(
           name="habitat_two_step",
           stages=(
               Stage(
                   "extract_voxel_features",
                   Spec("raw", {"modalities": list(modalities)}),
               ),
               Stage(
                   "preprocess1",
                   Spec(
                       "winsorize",
                       {
                           "winsor_limits": (0.05, 0.05),
                           "across_features": False,
                       },
                   ),
               ),
               Stage("preprocess2", Spec("minmax", {"across_features": False})),
               Stage(
                   "partition",
                   Spec(
                       "kmeans",
                       {"n_supervoxels": 50, "max_iter": 300, "n_init": 10},
                   ),
               ),
               Stage("pool", Spec("pool")),
               Stage(
                   "preprocess3",
                   Spec(
                       "binning",
                       {
                           "n_bins": 10,
                           "bin_strategy": "uniform",
                           "across_features": False,
                       },
                   ),
               ),
               Stage(
                   "fit",
                   Spec(
                       "kmeans",
                       {
                           "min_habitats": 2,
                           "max_habitats": 10,
                           "validation": "elbow",
                           "max_iter": 300,
                           "n_init": 10,
                       },
                   ),
               ),
               Stage("assign", Spec("nearest_centroid")),
               # connected_components is not a plugin domain yet; role= is the
               # documented escape hatch (omit this stage to skip cleanup).
               Stage(
                   "postprocess_habitat",
                   Spec(
                       "connected_components",
                       {
                           "min_component_size": 100,
                           "connectivity": 1,
                           "reassign_method": "neighbor_vote",
                           "max_iterations": 3,
                       },
                   ),
                   role="postprocess_habitat",
               ),
           ),
           random_seed=42,
       )

       # Match YAML processes / timeout so process-pool execution matches CLI.
       policy = RunPolicy(
           workers=2,
           backend="process",
           subject_timeout_sec=900.0,
           resume=False,
       )
       backend = backend_from_policy(policy)

       result = recipes.Study(spec=spec).fit_predict(cohort, backend=backend)
       out_dir = Path("demo_data/results/habitat_two_step")
       result.save(
           out_dir,
           write_maps=True,
           write_units_table=True,
           write_cluster_plots=True,
       )
       print(result.habitat_model.summary())

       # Export a complete effective v1 YAML so CLI / run_from_yaml can replay
       # this exact run (expanded defaults, not only overridden fields).
       from habit import save_habitat_config

       save_habitat_config(
           "demo_data/results/habitat_two_step/effective_config.yaml",
           spec,
           data_source="demo_data/preprocessed",
           out_dir=out_dir,
           policy=policy,
       )


   if __name__ == "__main__":
       main()

Outputs land under ``demo_data/results/habitat_two_step/`` (including
``*_habitats.nrrd`` and ``habitat_model.habitatmodel``). The exported
``effective_config.yaml`` is a native v1 document: reload it with
:func:`~habit.recipes.run_from_yaml` or
``habit get-habitat --config …/effective_config.yaml`` for **voxel-identical**
habitat maps (same seed, data, and policy).

The overlay below is **not** from ``main()`` above. ``main()`` matches the
CLI YAML (four modalities, ``roi="pre_contrast"``) and draws a different
map. This PNG is written by the two-step gallery
(:doc:`../examples/two_step_habitat`) — Script + Draw the figures. Reproduce
it with one line (needs ``[viz]``)::

   python docs/source/examples/scripts/two_step_habitat_quickstart.py

The plot call in that script (``ROI = "LAP"``)::

   fig = plot_habitat_overlay(subject.image(ROI), habitat_map, title="habitats")

.. figure:: ../_static/images/examples/two_step_overlay.png
   :alt: Demo subj001 LAP with two-step habitat labels overlaid
   :width: 480

   Same file the gallery script writes to ``out/two_step_overlay.png``.

Further notebook-oriented patterns (synthetic cohorts, custom extractors)
live under :doc:`../examples/index`.

2. View
-------

CLI twin (one line — conda / Windows terminals do not support ``\`` continuation)::

   habit view demo_data/preprocessed/images/subj001/LAP/WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd demo_data/results/habitat_two_step/subj001_habitats.nrrd

Append inside the same ``main()`` from step 1 (uses ``cohort`` /
``result``)::

   from habit.viz import view_habitat_napari

   # Anatomy under LAP. Do not pass direction=volume.direction — that image
   # header can flip coronal/sagittal relative to the ROI / HabitatMap.
   volume = cohort[0].image("LAP")
   view_habitat_napari(volume, result.habitat_maps[0])

Needs napari (:doc:`installation`). Blocks until you close the window.
In napari, select the habitats Labels layer (Contour ``0`` = filled regions).

For fuller 3D review, also open the source volume and ``*_habitats.nrrd``
in **ITK-SNAP**, **3D Slicer**, or a **SimpleITK**-based viewer — not only
the napari 2D slice slider.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. figure:: ../_static/images/habitat_view_napari_region.png
          :alt: napari habitat view with filled region labels
          :width: 100%

          Filled labels.

     - .. figure:: ../_static/images/habitat_view_napari_contour.png
          :alt: napari habitat view with contour outlines
          :width: 100%

          Contour outlines.

3. Apply a saved model
----------------------

After step 1, the archive sits at
``demo_data/results/habitat_two_step/habitat_model.habitatmodel``.
CLI twin::

   habit get-habitat --config config/habitat/config_habitat_two_step_predict.yaml -m predict

Still inside ``main()`` (reuses ``cohort``, ``spec``, ``backend``)::

   from habit import HabitatModel

   model = HabitatModel.load(
       "demo_data/results/habitat_two_step/habitat_model.habitatmodel"
   )
   # Reuse the same HabitatSpec as training (feature chains must match).
   prediction = recipes.Study.from_model(model, spec).predict(
       cohort, backend=backend
   )
   prediction.save("demo_data/results/habitat_two_step/predict")
   print(len(prediction.habitat_maps), "subjects labelled")

4. Extract habitat features
---------------------------

CLI twin::

   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

Needs step 1 outputs (``*_habitats.nrrd``). Pass the same fields as the
demo YAML, as a Python ``dict`` (still no config file).
``n_processes: 2`` also spawns workers — keep this call under
``if __name__ == "__main__":`` (see the note in step 1).

.. code-block:: python

   def main() -> None:
       extract_result = recipes.extract_habitat_features(
           {
               "raw_img_folder": "demo_data/preprocessed",
               "habitats_map_folder": "demo_data/results/habitat_two_step",
               "out_dir": "demo_data/results/features",
               "n_processes": 2,
               "habitat_pattern": "*_habitats.nrrd",
               "feature_types": [
                   "volume",
                   "msi",
                   "ith_score",
                   "non_radiomics",
                   # Built-in graph topology (opt-in):
                   # "graph",
                   # Heavy PyRadiomics (opt-in):
                   # "traditional",
                   # "whole_habitat",
                   # "each_habitat",
               ],
           }
       )
       print(extract_result.output_dir)


   if __name__ == "__main__":
       main()

Uncomment ``"graph"`` (or call
:func:`~habit.extract_graph_features`) to also extract habitat graph topology
features — see :doc:`../how_to/graph_features` and
:doc:`../examples/graph_features`.

Voxel **texture** maps (local entropy on anatomy + ROI) are a separate
``habit.viz`` path — see :doc:`../how_to/voxel_texture` and
:doc:`../examples/voxel_texture`.

5. Tabular ML
-------------

CLI twin::

   habit model --config config/machine_learning/config_machine_learning_radiomics_minimal.yaml --mode train

Needs the **ML pack** under ``demo_data/ml_data/`` (see :doc:`quickstart`
step 2). Build a :class:`~habit.contracts.FeatureTable` and
:class:`~habit.MLSpec` in code (no process pool; ``__main__`` optional
here, but harmless to keep)::

   from pathlib import Path

   import pandas as pd

   from habit import FeatureTable, MLSpec, Spec
   from habit.contracts.outcome import BinaryOutcome
   from habit.contracts.provenance import Provenance
   import habit.recipes as recipes

   csv_path = Path("demo_data/ml_data/breast_cancer_dataset.csv")
   frame = pd.read_csv(csv_path, dtype={"subject_id": str})
   feature_columns = tuple(
       column
       for column in frame.columns
       if column not in {"subject_id", "label"}
   )
   # Prefix matches the YAML ``input.name: radiomics_``
   renamed = {
       column: f"radiomics_{column}" for column in feature_columns
   }
   table_frame = frame.rename(columns=renamed)
   table = FeatureTable(
       frame=table_frame,
       id_columns=("subject_id",),
       feature_columns=tuple(renamed[column] for column in feature_columns),
       outcome=BinaryOutcome(column="label", positive_label=1),
       provenance=Provenance.source("quickstart_python"),
   )

   # Mirrors config_machine_learning_radiomics_minimal.yaml → MLSpec
   # (variance before z-score; correlation after).
   ml_spec = MLSpec(
       name="ml_model",
       steps=(
           Spec("variance", {"threshold": 0.2}),
           Spec("zscore"),
           Spec("correlation", {"threshold": 0.8, "method": "spearman"}),
       ),
       classifier=Spec(
           "LogisticRegression",
           {
               "C": 1.0,
               "penalty": "l2",
               "solver": "lbfgs",
               "max_iter": 1000,
           },
       ),
       random_seed=42,
   )

   train_ids = Path("demo_data/ml_data/train_ids.txt").read_text(
       encoding="utf-8"
   ).split()
   test_ids = Path("demo_data/ml_data/test_ids.txt").read_text(
       encoding="utf-8"
   ).split()

   ml_result = recipes.train_model(
       table,
       ml_spec,
       seed=42,
       train_ids=train_ids,
       test_ids=test_ids,
   )
   print(ml_result.test_metrics)

Next
----

* Examples gallery: :doc:`../examples/index`
* Graph topology / voxel texture: :doc:`../how_to/graph_features` ·
  :doc:`../how_to/voxel_texture`
* API reference: :doc:`../api/index`
* Your own data: :doc:`../how_to/prepare_data`
