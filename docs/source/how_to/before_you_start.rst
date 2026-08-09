Before you start
================

Do this once, then every how-to page is just copy → edit ★ → run.
A git clone is **not** required after ``pip install habitat-analysis``.

1. Terminal + env
-----------------

Open a **conda** terminal first. On Windows: Start (Win10 often
bottom-left; Win11 often bottom-center) → **Anaconda3** → **Anaconda
Prompt**, or search ``Anaconda Prompt`` — not plain CMD/PowerShell.
Details and screenshots: :doc:`../tutorial/installation`.

::

   conda activate habit          # prompt must show (habit)
   habit --version

2. Work directory + demo configs
--------------------------------

Pick any folder you own as ``<work_dir>``. Materialize the bundled demo
YAML tree (shipped inside the wheel; not ``demo_data``)::

   mkdir D:\my_habit_work        # Windows — use your path
   cd D:\my_habit_work
   habit copy-demo-config --dest .

   # macOS / Linux
   mkdir -p ~/my_habit_work && cd ~/my_habit_work
   habit copy-demo-config --dest .

   ls config                     # or: Test-Path config

Python::

   from habit import copy_demo_config
   copy_demo_config(r"D:/my_habit_work")

Commands below assume your shell ``cwd`` is this ``<work_dir>``.

3. Demo data (first run)
------------------------

Packs are **split** — habitat-only users need imaging; add ML only for
``habit model`` / ``habit cv``. Download into **your** ``<work_dir>``
(next to ``config/``), not into the Python package directory.

**Imaging** (``preprocessed.zip``):

* |download_demo_data| — extract code: |demo_data_code|
* Extract so you have ``demo_data/preprocessed/images/`` and
  ``demo_data/preprocessed/masks/`` next to ``config/``
* **No** nested ``processed_images`` under ``preprocessed/``
* Zip layout tips:

  - top-level ``preprocessed/`` → extract into ``demo_data/``
  - top-level ``images/`` + ``masks/`` → put under ``demo_data/preprocessed/``

* Modalities: ``pre_contrast`` / ``LAP`` / ``PVP`` / ``delay_3min``
* Preprocessed tree is already there — skip preprocess the first time

**Tabular ML** (``ml_data.zip``, optional):

* |download_ml_data| — extract code: |ml_data_code|
* Extract to ``demo_data/ml_data/`` (e.g. ``breast_cancer_dataset.csv``)
* If zip top level is ``ml_data/``, extract into ``demo_data/``

4. Paths in YAML
----------------

* **v0.1** configs (most files): relative paths resolve from the **YAML
  file's directory** (hence ``../../demo_data/...``).
* **v1** (``version: '1.0'`` / ``*_v1.yaml``): paths as written; run from
  ``<work_dir>`` or use absolute paths.
* Prefer ``D:/data/...``; quote only if the path has spaces.

5. Safe YAML edits
------------------

* Spaces for indent (no Tab); ``key: value``; lowercase ``true`` / ``false``
* First run: change only ★ **MUST EDIT** fields

Validate without running::

   habit check-config --config config/habitat/config_habitat_two_step.yaml

Next: :doc:`prepare_data`.
