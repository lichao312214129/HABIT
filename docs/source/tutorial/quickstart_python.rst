Quickstart: Python API
========================

Install first (:doc:`installation`). This is the **beginner** Python
path: construct a cohort, :class:`~habit.spec.HabitatSpec`, and call
:mod:`habit.recipes` with explicit arguments (no YAML, no enclosing
functions). Every line runs directly in a Jupyter Notebook, an IPython
session, or a plain ``.py`` script.

To embed **one** operator in your own notebook or pipeline (no
``Study``), see :doc:`../examples/habitat_atomic_ops` after this page.
Concepts: :doc:`habitat_analysis`.

Get the official imaging pack once (prints the path and the folder tree
your own data must match). See :doc:`quickstart` step 2::

   # Windows - Anaconda Prompt
   conda activate habit
   habit fetch-demo

   # same thing in Python:
   from habit.datasets import fetch_demo
   DATA = fetch_demo()

1. Habitat analysis (two-step)
------------------------------

Run the code below directly. No ``def main():`` required:

.. code-block:: python

   from pathlib import Path

   from habit.contracts import cohort_from_directory
   from habit.datasets import fetch_demo
   from habit.spec import HabitatSpec, Spec, Stage
   import habit.recipes as recipes

   # 1. Load data
   DATA = fetch_demo()
   modalities = ("pre_contrast", "LAP", "PVP", "delay_3min")
   # ROI key is the first modality in the concat list.
   cohort = cohort_from_directory(
       DATA,
       modalities=modalities,
       roi="pre_contrast",
   )
   print(cohort)

   # 2. Declare stages (partition + pool => two_step design)
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

   # 3. Fit and predict
   result = recipes.Study(spec=spec).fit_predict(cohort)
   out_dir = Path("demo_data/results/habitat_two_step")
   result.save(
       out_dir,
       write_maps=True,
       write_units_table=True,
       write_cluster_plots=True,
   )
   print(result.habitat_model.summary())

Outputs land under ``demo_data/results/habitat_two_step/`` (including
``*_habitats.nrrd`` and ``habitat_model.habitatmodel``).

.. figure:: ../_static/images/examples/two_step_overlay.png
   :alt: Demo subj001 LAP with two-step habitat labels overlaid
   :width: 480

   Two-step habitat overlay (``subj001`` LAP).

2. View
-------

Directly visualize the result generated above (uses ``cohort`` and ``result``):

.. code-block:: python

   from habit.viz import view_habitat_napari

   volume = cohort[0].image("LAP")
   view_habitat_napari(volume, result.habitat_maps[0])

Needs napari (:doc:`installation`). Blocks until you close the window.
In napari, select the habitats Labels layer (Contour ``0`` = filled regions).

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

After step 1, the fitted model sits at
``demo_data/results/habitat_two_step/habitat_model.habitatmodel``.
Load it and apply it to new data (or the same cohort):

.. code-block:: python

   from habit.contracts import HabitatModel

   model = HabitatModel.load(
       "demo_data/results/habitat_two_step/habitat_model.habitatmodel"
   )
   prediction = recipes.Study.from_model(model, spec).predict(cohort)
   prediction.save("demo_data/results/habitat_two_step/predict")
   print(len(prediction.habitat_maps), "subjects labelled")

4. Extract habitat features
---------------------------

Quantify the segmented habitats with volume, spatial interaction (MSI),
intratumoural heterogeneity (ITH), and graph topology:

.. code-block:: python

   extract_result = recipes.extract_habitat_features(
       {
           "raw_img_folder": "demo_data/preprocessed",
           "habitats_map_folder": "demo_data/results/habitat_two_step",
           "out_dir": "demo_data/results/features",
           "n_processes": 1,
           "habitat_pattern": "*_habitats.nrrd",
           "feature_types": [
               "volume",
               "msi",
               "ith_score",
               "non_radiomics",
               "graph",
           ],
       }
   )
   print("Features extracted to:", extract_result.output_dir)

``"graph"`` extracts graph network features (or call
:func:`~habit.kernels.extract_graph_features` on a label array). See
:doc:`../how_to/graph_features` and :doc:`../examples/graph_features`.

.. note::

   **Scaling to multiple processes**:
   When running large cohorts in batch mode with a process pool
   (e.g., ``RunPolicy(workers=4, backend="process")``), wrap execution in
   ``if __name__ == "__main__":`` to satisfy Windows multiprocessing spawn
   guards. See :doc:`execution` for details.

Next
----

* Habitat analysis strategies: :doc:`habitat_analysis`
* Embed one operator in your workflow: :doc:`../examples/habitat_atomic_ops`
* Parallel execution and fault tolerance: :doc:`execution`
* Habitat Guide: :doc:`../examples/index`
* API reference: :doc:`../api/index`
* Your own data: :doc:`../examples/data_from_arrays`
* YAML / CLI demo: :doc:`quickstart`
