Before you start
================

Do this once, then every CLI bookmark is copy → edit ★ → run.
A git clone is **not** required after ``pip install habitat-analysis``.

Habitat analysis is the core (Guide: :doc:`../examples/index`).
These steps only set up the terminal, ``config/``, and the demo imaging pack.

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

   from habit.api.demo_config import copy_demo_config
   copy_demo_config(r"D:/my_habit_work")

Commands below assume your shell ``cwd`` is this ``<work_dir>``.

3. Demo data (first run)
------------------------

Packs are **split** — habitat-only users need imaging; add ML only for
``habit model`` / ``habit cv``. Fetch imaging **once**; HABIT prints the
path and the folder tree (that tree is the contract for your own data).

**Imaging** — from ``<work_dir>``::

   habit fetch-demo --work-dir .

   # same thing in Python
   from habit.datasets import fetch_demo
   DATA = fetch_demo()          # prints DATA, subjects, series, example files

* Cache: ``~/.habit_data/demo-data-v1/preprocessed`` (override with
  ``HABIT_DATA``). Later calls do not download again.
* ``--work-dir .`` links ``<work_dir>/demo_data/preprocessed`` to that cache
  so shipped YAML paths keep working.
* Modalities in the pack: ``pre_contrast`` / ``LAP`` / ``PVP`` /
  ``delay_3min``
* Preprocessed tree is already there — skip preprocess the first time
* Backup share if GitHub is blocked: |download_demo_data| (code
  |demo_data_code|)

**Tabular ML** (``ml_data.zip``, optional):

* |download_ml_data| — extract code: |ml_data_code|
* Extract to ``demo_data/ml_data/`` (e.g. ``breast_cancer_dataset.csv``)
* If zip top level is ``ml_data/``, extract into ``demo_data/``

4. Paths in YAML
----------------

* Most shipped configs: relative paths resolve from the **YAML file's
  directory** (hence ``../../demo_data/...``).
* Documents with ``version: '1.0'`` / ``*_v1.yaml``: paths as written; run
  from ``<work_dir>`` or use absolute paths.
* Prefer ``D:/data/...``; quote only if the path has spaces.

5. Safe YAML edits
------------------

* Spaces for indent (no Tab); ``key: value``; lowercase ``true`` / ``false``
* First run: change only ★ **MUST EDIT** fields

Validate without running::

   habit check-config --config config/habitat/config_habitat_two_step.yaml

Next: :doc:`../examples/data_from_arrays` — directory / SimpleITK / NumPy / CLI path-list YAML
(Option B), or the Python gallery load
(:func:`~habit.contracts.cohort_from_directory` with ``DATA`` / ``MODALITIES`` /
``ROI``, Option C). The same three knobs appear in every gallery script.
