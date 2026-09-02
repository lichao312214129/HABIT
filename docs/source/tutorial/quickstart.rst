Quickstart: run the demo (YAML + CLI)
======================================

No Python required. Install first (:doc:`installation`).
You do **not** need a git clone: ``pip install habitat-analysis`` is enough.

1. Work directory and demo configs
----------------------------------

Pick a folder you own (``<work_dir>``). In a conda terminal::

   # Windows — Anaconda Prompt (not plain CMD)
   conda activate habit          # prompt must show (habit)
   mkdir D:\my_habit_work
   cd D:\my_habit_work
   habit copy-demo-config --dest .

   # macOS / Linux
   conda activate habit
   mkdir -p ~/my_habit_work && cd ~/my_habit_work
   habit copy-demo-config --dest .

This writes ``<work_dir>/config/``. ``demo_data/`` is **not** in the wheel;
download it next.

2. Get demo data
----------------

Packs are **split**. Habitat / view / extract need only the imaging pack.
Download the ML pack only if you run ``habit model``.

**Imaging** — |download_demo_data| — extract code: |demo_data_code|

Download ``preprocessed.zip`` and extract inside ``<work_dir>`` so you have::

   <work_dir>/
   ├── config/                 # from habit copy-demo-config
   └── demo_data/
       └── preprocessed/
           ├── images/
           └── masks/

There is **no** nested ``processed_images`` layer under ``preprocessed/``.

* Zip top level is ``preprocessed/`` → extract into ``demo_data/``.
* Zip top level is ``images/`` and ``masks/`` → put them under
  ``demo_data/preprocessed/``.

Preprocessed images are already included — skip preprocess on the first run.

**Tabular ML** (only for ``habit model``) — |download_ml_data| — extract
code: |ml_data_code|

Download ``ml_data.zip`` and extract so ``demo_data/ml_data/`` sits next to
``demo_data/preprocessed/``. If the zip top level is ``ml_data/``, extract
into ``demo_data/``.

3. Activate and check
---------------------

Stay in the conda env from :doc:`installation`. From ``<work_dir>``::

   # Windows — Anaconda Prompt
   conda activate habit
   cd D:\my_habit_work

   # macOS / Linux
   conda activate habit
   cd ~/my_habit_work

   habit --version

4. Run
------

If ``get-habitat`` complains about parquet or plots, install
``habitat-analysis[tables,viz]`` (:doc:`installation`).

::

   habit check-config --config config/habitat/config_habitat_two_step.yaml
   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

   habit view demo_data/preprocessed/images/subj001/LAP/WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd demo_data/results/habitat_two_step/subj001_habitats.nrrd

   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

``habit extract`` uses the habitat maps from ``get-habitat`` (imaging pack).
``habit model`` needs ``demo_data/ml_data/`` from step 2::

   habit model --config config/machine_learning/config_machine_learning_radiomics_minimal.yaml --mode train

``habit view`` opens napari if installed; otherwise it writes a PNG. In
napari, select the habitats Labels layer (Contour ``0`` = filled regions).

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

Outputs land under ``demo_data/results/``.

Next
----

* Your own data: :doc:`../how_to/prepare_data` then :doc:`../how_to/index`
* Graph topology features: :doc:`../how_to/graph_features`
* Voxel texture maps: :doc:`../how_to/voxel_texture`
* 3D viewers (ITK-SNAP / 3D Slicer): load the source image and
  ``*_habitats.nrrd`` together
* Python API: :doc:`quickstart_python`
* All commands: :doc:`../reference/cli`
