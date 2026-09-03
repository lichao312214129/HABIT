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

This writes ``<work_dir>/config/``. Imaging data is **not** in the wheel;
fetch it next (once).

2. Get demo data
----------------

From ``<work_dir>``::

   habit fetch-demo --work-dir .

The first call downloads the official 5-subject preprocessed pack (about
473 MB) into ``%USERPROFILE%\.habit_data\demo-data-v1\preprocessed``
(or ``$HOME/.habit_data/...``). Later calls reuse that cache. The command
prints the absolute path and the folder tree — that tree is what **your**
data must look like (same ``images/<id>/<series>/`` +
``masks/<id>/<roi>/`` layout; change IDs and series names).

``--work-dir .`` also creates ``<work_dir>/demo_data/preprocessed`` pointing
at the cache so the shipped YAML ``../../demo_data/preprocessed`` paths keep
working.

Preprocessed images are already included — skip preprocess on the first run.

If GitHub is unreachable, the imaging zip is also on the backup share
(|download_demo_data|, code |demo_data_code|). Extract so you have
``demo_data/preprocessed/images/`` and ``masks/``.

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

``habit extract`` uses the habitat maps from ``get-habitat`` to extract volume, MSI, ITH, and graph features.

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

* Habitat analysis (what / which strategy): :doc:`habitat_analysis`
* Your own data: :doc:`../examples/data_from_arrays`
* Python API (beginner ``Study``): :doc:`quickstart_python`
* Embed one operator: :doc:`../examples/habitat_atomic_ops`
* Parallel / fault tolerance: :doc:`execution`
* All commands: :doc:`../reference/cli`
