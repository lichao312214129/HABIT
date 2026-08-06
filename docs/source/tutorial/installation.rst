Installation
============

This page explains how to install HABIT. It is written for **two audiences**:

* **Clinicians / analysts who do not program** — prefer the Windows one-click
  ZIP (no Python knowledge required).
* **Developers / programmers** — prefer ``pip`` / ``conda`` so HABIT can live
  inside an existing scientific Python environment (notebooks, pipelines).

If you only want to run the demo with YAML + CLI, start with
**Path A**. If you want ``import habit`` in Python, use **Path B**.

Demo data after install: :doc:`quickstart` (YAML) or :doc:`quickstart_python`
(Python API).

Which path should I choose?
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Your situation
     - Recommended path
     - Why
   * - Windows PC, first-time user, mainly YAML + CLI
     - **Path A** — Windows lightweight ZIP
     - No need to install Python yourself; double-click installers
   * - macOS / Linux, or you already use Conda
     - **Path B** — Conda + pip
     - Same commands on all platforms; easy to update
   * - Embed HABIT in your own Python project / notebook
     - **Path B** — ``pip install habitat-analysis``
     - Library-first install; pick optional extras as needed
       (``import habit`` stays the same)
   * - Hospital PC with locked admin rights
     - Prefer **Path A** (portable folder) or ask IT for Conda
     - Path A does not modify system Python / PATH

Supported Python versions (Path B): **3.10, 3.11, 3.12**.
The Windows ZIP (Path A) pins **Python 3.10.20** for reproducibility.

.. note::

   On PyPI the project is published as **``habitat-analysis``**
   (``pip install habitat-analysis``). The import name stays ``import habit``.
   The bare name ``HABIT`` is not allowed by PyPI naming policy.


Path A — Windows lightweight installer (recommended for clinicians)
-------------------------------------------------------------------

You do **not** need to install Python or Conda beforehand.

Step-by-step
~~~~~~~~~~~~

1. **Download** the HABIT lightweight ZIP from the project release / website
   (the package that contains ``launchers/``).
2. **Extract** to a short local path with **ASCII letters only and no spaces**,
   for example::

      D:\HABIT

   Avoid ``C:\Users\张三\Desktop\HABIT 新版本`` — Chinese characters and spaces
   often break scientific tools on Windows.
3. Open the ``launchers`` folder.
4. Double-click ``一键安装HABIT.bat`` and wait until it finishes
   (first run needs network access; may take several minutes).
5. Double-click ``启动HABIT命令行.bat`` — a terminal opens that is already
   configured for this HABIT copy.
6. In that terminal, type and press Enter::

      habit --version

   You should see something like ``HABIT, version 1.0.0``.

Optional add-ons (only if you need them)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Still inside ``launchers/``:

* ``一键启用HABIT-AutoML.bat`` — AutoGluon tabular models
* ``一键启用HABIT-进阶分析.bat`` — SHAP, Plotly, ICC helpers, survival tools
* ``一键启用HABIT-GPU.bat`` — NVIDIA GPU acceleration (optional; if it fails,
  the CPU install remains valid)

What the installer does (for your IT department)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Creates an isolated Python 3.10 environment under the extracted folder
  (project-managed ``.mamba`` / env prefix).
* Does **not** require admin rights for a normal user-writable drive.
* Does **not** permanently change the system ``PATH``.
* Validates path, disk space, and package assets before changing the env.
* Installs a prebuilt PyRadiomics wheel from ``installer/vendor/`` when that
  asset is present in the release bundle.

If installation fails: :doc:`../troubleshooting/faq`.


Path B — pip / Conda install (developers and cross-platform)
------------------------------------------------------------

Use this path when you want HABIT as a normal Python package.

Prerequisites
~~~~~~~~~~~~~

1. Install **Miniconda** or **Anaconda** (recommended), *or* a standalone
   Python 3.10–3.12 from https://www.python.org .
2. Open a terminal:

   * Windows: **Anaconda Prompt** or PowerShell
   * macOS / Linux: Terminal

3. Confirm Python::

      python --version

   Expect ``Python 3.10.x``, ``3.11.x``, or ``3.12.x``.

B1. Create an isolated environment (strongly recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy-paste (Conda)::

   conda create -n habit python=3.10 -y
   conda activate habit

Using a dedicated environment avoids conflicts with other hospital / lab
software.

B2. Install HABIT with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~

**From PyPI** (when the release is published)::

   pip install -U pip
   pip install habitat-analysis
   habit --version

**From a local source checkout** (developers)::

   pip install -U pip
   cd /path/to/habit_project
   pip install .
   habit --version

**Editable / development install** (you change HABIT source frequently)::

   cd /path/to/habit_project
   pip install -e .
   habit --version

What the base install includes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default ``pip install habitat-analysis`` is intentionally **light**:

* Image I/O and geometry helpers
* Image preprocessing recipes
* Habitat analysis with intensity / raw-concat style features
* Core tabular machine learning
* CLI entry point ``habit``

It does **not** pull PyRadiomics by default (see next section).

Verify the install
~~~~~~~~~~~~~~~~~~

Run these three checks::

   habit --version
   python -c "import habit; print(habit.__version__)"
   python -c "import habit.recipes; print('recipes OK')"

All three should succeed without error.


Install PyRadiomics (only if you need radiomics features)
---------------------------------------------------------

You need PyRadiomics when your YAML / API uses:

* ``voxel_radiomics`` / ``supervoxel_radiomics``
* ``habit radiomics`` (traditional radiomics)
* habitat feature extractors that call PyRadiomics textures

If you only use intensity ``raw`` / ``concat`` habitat features, you can skip
this section.

Option 1 — Conda-forge (preferred when Conda works)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the ``habit`` environment activated::

   conda install -c conda-forge pyradiomics -y
   python -c "import radiomics; print('pyradiomics OK')"

This is the most portable route across Python 3.10–3.12 **when your Conda
installation can write to its package cache**.

Option 2 — pip with a **pinned** wheel (verified on Windows Python 3.10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Upstream PyPI packaging for PyRadiomics is fragile:

* ``pip install pyradiomics`` (no pin) often downloads **3.1.0 sdist**, which
  fails with ``No module named 'versioneer'``
* **Do not** rely on ``habitat-analysis[radiomics]`` alone against an unpinned
  3.1.0 — that was verified to fail on a clean Windows 3.10 venv
* **Does work** (verified locally): pin **3.0.1**, which still has a Windows
  ``cp310`` binary wheel

Recommended pip recipe::

   pip install habitat-analysis
   pip install "pyradiomics==3.0.1"
   python -c "import radiomics, habit; print(habit.__version__, 'pyradiomics OK')"

From habitat-analysis ≥ 1.0.1 the ``[radiomics]`` extra also pins
``pyradiomics>=3.0.1,<3.1``, so this is equivalent::

   pip install "habitat-analysis[radiomics]"

If pip still tries to build from source and fails, force the binary wheel::

   pip install "pyradiomics==3.0.1" --only-binary=:all:

On **Python 3.12+**, prefer Option 1 (conda-forge); PyPI wheels for 3.0.1 may
not exist for that ABI.

Windows ZIP note
~~~~~~~~~~~~~~~~

Path A already installs a checked ``cp310`` wheel from
``installer/vendor/pyradiomics-3.0.1-cp310-cp310-win_amd64.whl`` when the
release bundle includes that file — you usually do not need a separate
PyRadiomics step after the one-click installer.


Optional extras (programmers)
-----------------------------

Install only what your workflow needs. Missing extras raise
``OptionalDependencyError`` with the install hint.

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Extra
     - Command
     - Needed for
   * - ``radiomics``
     - ``pip install "pyradiomics==3.0.1"`` then
       ``pip install "habitat-analysis[radiomics]"`` (≥ 1.0.1 pins ``<3.1``)
     - Voxel / supervoxel / traditional PyRadiomics
   * - ``ml``
     - ``pip install "habitat-analysis[ml]"``
     - XGBoost, SMOTE, mRMR / VIF / stepwise selectors
   * - ``registration``
     - ``pip install "habitat-analysis[registration]"``
     - ANTs registration backend in preprocessing
   * - ``analysis``
     - ``pip install "habitat-analysis[analysis]"``
     - SHAP, Plotly, ICC (pingouin), survival tools
   * - ``automl``
     - ``pip install "habitat-analysis[automl]"``
     - AutoGluon Tabular
   * - ``torch``
     - ``pip install "habitat-analysis[torch]"``
     - TorchRadiomics / GPU texture backends
   * - ``gui``
     - ``pip install "habitat-analysis[gui]"``
     - Web GUI server dependencies (preview)
   * - ``all``
     - ``pip install "habitat-analysis[all]"``
     - Everything above (large download)

Combine extras::

   pip install "habitat-analysis[ml,analysis]"

From a source checkout (editable)::

   pip install -e ".[ml,analysis]"


Copy-paste recipes
------------------

Clinician on Windows (Path A)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Extract ZIP to ``D:\HABIT``
2. Run ``launchers\一键安装HABIT.bat``
3. Run ``launchers\启动HABIT命令行.bat``
4. ``habit --version``
5. Follow :doc:`quickstart`

Developer on any OS (minimal library)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   conda create -n habit python=3.10 -y
   conda activate habit
   pip install -U pip
   pip install habitat-analysis
   habit --version

Developer who needs radiomics + ML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   conda create -n habit python=3.10 -y
   conda activate habit
   conda install -c conda-forge pyradiomics -y
   pip install -U pip
   pip install "habitat-analysis[ml,analysis]"
   habit --version
   python -c "import radiomics, habit; print(habit.__version__, 'OK')"

From GitHub source
~~~~~~~~~~~~~~~~~~

::

   git clone https://github.com/lichao312214129/HABIT.git
   cd HABIT
   conda create -n habit python=3.10 -y
   conda activate habit
   pip install -U pip
   pip install .
   # optional:
   conda install -c conda-forge pyradiomics -y
   pip install -e ".[ml]"


Common problems
---------------

**``habit`` is not recognized**

* Path A: open ``启动HABIT命令行.bat`` first (do not use a random system
  terminal).
* Path B: ``conda activate habit``, then retry. Confirm with
  ``where habit`` (Windows) or ``which habit`` (macOS/Linux).

**``pip install habitat-analysis`` fails mentioning ``pyradiomics``**

Base HABIT no longer requires PyRadiomics. Upgrade to HABIT ≥ 1.0.0 packaging
that declares the ``radiomics`` extra, or install without extras first::

   pip install habitat-analysis

**``pip install pyradiomics`` fails on Python 3.12**

Expected with current upstream PyPI packages. Use::

   conda install -c conda-forge pyradiomics

or create a **Python 3.10 / 3.11** environment.

**Permission / SSL / mirror errors at hospital networks**

Ask IT to allow access to ``pypi.org`` and ``conda-forge``, or use an internal
mirror. Offline Path A ZIP avoids most of these issues.

**More FAQ**

See :doc:`../troubleshooting/faq`.


Next steps
----------

* Clinicians (YAML + CLI, no programming): :doc:`quickstart`
* Programmers (Python API): :doc:`quickstart_python`
* Examples gallery: :doc:`../examples/index`
