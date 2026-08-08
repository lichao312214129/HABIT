Before you start
================

Do this once, then every how-to page is just copy → edit ★ → run.

1. Terminal + env
-----------------

Open a **conda** terminal (Windows: **Anaconda Prompt**), activate the
env, then confirm the CLI. Details and ASCII guide:
:doc:`../tutorial/installation`.

::

   conda activate habit          # prompt must show (habit)
   habit --version

2. Project root
---------------

Commands run from the folder that contains ``config/`` (and ``demo_data/``
after you unpack the demo)::

   cd /d D:\habit          # Windows — use your path
   cd ~/habit              # macOS / Linux
   ls config               # or: Test-Path config

3. Demo data (first run)
------------------------

Packs are **split** — habitat-only users need imaging; add ML only for
``habit model`` / ``habit cv``.

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
  project root or use absolute paths.
* Prefer ``D:/data/...``; quote only if the path has spaces.

5. Safe YAML edits
------------------

* Spaces for indent (no Tab); ``key: value``; lowercase ``true`` / ``false``
* First run: change only ★ **MUST EDIT** fields

Validate without running::

   habit check-config --config config/habitat/config_habitat_two_step.yaml

Next: :doc:`prepare_data`.
