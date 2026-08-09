Quickstart: run the demo (YAML + CLI)
======================================

No Python required. Install HABIT first (:doc:`installation`).
You do **not** need a git clone: ``pip install habitat-analysis`` is enough.

1. Create a work directory and copy demo configs
------------------------------------------------

Pick any folder you own (called ``<work_dir>`` below). Activate conda, then
materialize the bundled demo YAML tree into ``<work_dir>/config/``::

   # Windows — Anaconda Prompt (not plain CMD)
   conda activate habit          # prompt must show (habit)
   mkdir D:\my_habit_work
   cd D:\my_habit_work
   habit copy-demo-config --dest .

   # macOS / Linux
   conda activate habit
   mkdir -p ~/my_habit_work && cd ~/my_habit_work
   habit copy-demo-config --dest .

Python equivalent::

   from habit import copy_demo_config
   copy_demo_config(r"D:/my_habit_work")   # creates .../config/

``demo_data/`` is **not** inside the wheel; download it next (step 2).

2. Get demo data
----------------

Demo packs are **split**. Habitat / imaging steps need only the imaging
pack; download the ML pack only if you run ``habit model`` / ``habit cv``.

**Imaging** — |download_demo_data| — extract code: |demo_data_code|

Download ``preprocessed.zip``, then extract inside ``<work_dir>`` so you have::

   <work_dir>/
   ├── config/                 # from habit copy-demo-config
   └── demo_data/
       └── preprocessed/
           ├── images/
           └── masks/

There is **no** nested ``processed_images`` layer under ``preprocessed/``.

* If the zip top level is a ``preprocessed/`` folder, extract into
  ``demo_data/`` (result: ``demo_data/preprocessed/...``).
* If the zip top level is ``images/`` and ``masks/``, put them under
  ``demo_data/preprocessed/``.

Modalities: ``pre_contrast`` / ``LAP`` / ``PVP`` / ``delay_3min``.
Preprocessed images are already included — you can skip preprocess on the
first run.

**Tabular ML** (optional) — |download_ml_data| — extract code: |ml_data_code|

Download ``ml_data.zip`` and extract so ``demo_data/ml_data/`` sits next to
``demo_data/preprocessed/`` (CSV tables such as
``breast_cancer_dataset.csv``). If the zip top level is ``ml_data/``,
extract into ``demo_data/``.

3. Open a conda terminal in ``<work_dir>``
------------------------------------------

HABIT must run inside the activated conda env. Full guide:
:doc:`installation`.

**Find the conda terminal (Windows):**

1. Click **Start** (Windows logo) or press the Windows key.
   Win10: often **bottom-left**; Win11: often **bottom-center**.
2. Start → **Anaconda3** → **Anaconda Prompt** (or **Anaconda PowerShell
   Prompt** / Miniconda Prompt). Or search ``Anaconda Prompt``.
3. Do **not** use plain Command Prompt / PowerShell.

.. figure:: ../_static/images/open_anaconda_prompt_windows.png
   :alt: Windows Start menu: open Anaconda Prompt
   :width: 80%

   Open **Anaconda Prompt** from the Start menu (Win11 Start icon may be
   centered). Details: :doc:`installation`.

Then activate, ``cd`` to your ``<work_dir>`` (folder with ``config/``), and
check::

   # Windows — Anaconda Prompt (not plain CMD)
   conda activate habit          # prompt must show (habit)
   cd D:\my_habit_work        # your work_dir (has config/)

   # macOS / Linux — Terminal with conda available
   conda activate habit
   cd ~/my_habit_work

   habit --version

4. Run
------

If you followed :doc:`installation`, ``[tables,viz]`` is already installed.
Otherwise install them before ``get-habitat`` (demo YAML defaults to parquet
results and clustering curves)::

   pip install "habitat-analysis[tables,viz]" -i https://pypi.org/simple

::

   habit check-config --config config/habitat/config_habitat_two_step.yaml
   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

   habit view demo_data/preprocessed/images/subj001/LAP/WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd demo_data/results/habitat_two_step/subj001_habitats.nrrd

   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

The last command needs the **ML pack** (``ml_data.zip``). If you
skipped it in step 2: |download_ml_data| — extract code: |ml_data_code|.
Extract so ``demo_data/ml_data/`` sits next to ``demo_data/preprocessed/``
under ``<work_dir>`` (if the zip top level is ``ml_data/``, extract into
``demo_data/``)::

   habit model --config config/machine_learning/config_machine_learning_radiomics_minimal.yaml --mode train

``habit view`` opens napari if installed (see :doc:`installation`); otherwise
it falls back to a PNG. In napari, select the habitats Labels layer
(Contour ``0`` = filled regions).

For fuller 3D inspection, load the source image and ``*_habitats.nrrd``
together in **ITK-SNAP**, **3D Slicer**, or a **SimpleITK**-based viewer
(label overlay / segmentation), not only the napari 2D slice slider.

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
* Python API: :doc:`quickstart_python`
* All commands: :doc:`../reference/cli`
