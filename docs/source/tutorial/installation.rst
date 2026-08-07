Installation
============

Install HABIT with one of these two methods:

* **(A) pip** into a Miniconda / venv / existing Python environment
* **(B) from Git source** (clone or local checkout)

On PyPI the project is published as **``habitat-analysis``**
(``pip install habitat-analysis``). The import name stays ``import habit``.
The bare name ``HABIT`` is not allowed by PyPI naming policy.

Supported Python versions: **3.10, 3.11, 3.12, 3.13, 3.14**
(verified on Windows x64; both ``numpy`` 1.26 and 2.x are supported).

Demo data after install: :doc:`quickstart` (YAML) or :doc:`quickstart_python`
(Python API).


Method A — pip install
----------------------

Prerequisites
~~~~~~~~~~~~~

1. Install **Miniconda** / **Anaconda** (recommended), or a standalone
   Python 3.10–3.14 from https://www.python.org .
2. Open a terminal (Anaconda Prompt / PowerShell on Windows; Terminal on
   macOS / Linux).
3. Confirm Python::

      python --version

Create an isolated environment (strongly recommended)::

   conda create -n habit python=3.10 -y
   conda activate habit

Install from PyPI::

   pip install -U pip
   pip install habitat-analysis
   habit --version

What the base install includes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default ``pip install habitat-analysis`` is intentionally **light**:

* Image I/O and geometry helpers
* Image preprocessing recipes
* Habitat analysis with intensity / raw-concat style features
* Core tabular machine learning
* CLI entry point ``habit``

It does **not** pull PyRadiomics by default (see below).

Verify the install
~~~~~~~~~~~~~~~~~~

::

   habit --version
   python -c "import habit; print(habit.__version__)"
   python -c "import habit.recipes; print('recipes OK')"


Method B — install from Git source
----------------------------------

Use this when you develop HABIT or need an unreleased commit::

   git clone https://github.com/lichao312214129/HABIT.git
   cd HABIT
   conda create -n habit python=3.10 -y
   conda activate habit
   pip install -U pip
   pip install .
   habit --version

Editable install (you change HABIT source frequently)::

   pip install -e .
   habit --version

A local source checkout can also be installed with ``pip install .`` /
``pip install -e .`` without cloning from GitHub.


Install PyRadiomics (only if you need radiomics features)
---------------------------------------------------------

You need PyRadiomics when your YAML / API uses:

* ``voxel_radiomics`` / ``supervoxel_radiomics``
* ``habit radiomics`` (traditional radiomics)
* habitat feature extractors that call PyRadiomics textures

If you only use intensity ``raw`` / ``concat`` habitat features, you can skip
this section.

**Install PyRadiomics separately.** HABIT extras do **not** pull
``pyradiomics`` (PyPI rejects direct wheel URLs in uploaded metadata, and the
Windows PyPI sdist is broken).

Upstream PyPI has **no usable PyRadiomics Windows binaries** for any
supported Python (3.10–3.14): bare ``pip install pyradiomics`` downloads a
broken **3.1.0 sdist**. HABIT publishes self-built **PyRadiomics 3.1.0**
``win_amd64`` wheels (CPython 3.10–3.14) as assets of GitHub Release
`v1.0.2 <https://github.com/lichao312214129/HABIT/releases/tag/v1.0.2>`_.

Windows — prebuilt wheel (pick your Python)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   # Python 3.10
   pip install https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp310-cp310-win_amd64.whl

   # Python 3.11
   pip install https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp311-cp311-win_amd64.whl

   # Python 3.12
   pip install https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp312-cp312-win_amd64.whl

   # Python 3.13
   pip install https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp313-cp313-win_amd64.whl

   # Python 3.14
   pip install https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp314-cp314-win_amd64.whl

   python -c "import radiomics; print(radiomics.__version__)"

.. list-table:: Windows wheel URLs (Release ``v1.0.2``)
   :header-rows: 1
   :widths: 18 82

   * - Python
     - Wheel URL
   * - 3.10
     - https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp310-cp310-win_amd64.whl
   * - 3.11
     - https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp311-cp311-win_amd64.whl
   * - 3.12
     - https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp312-cp312-win_amd64.whl
   * - 3.13
     - https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp313-cp313-win_amd64.whl
   * - 3.14
     - https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp314-cp314-win_amd64.whl

macOS / Linux
~~~~~~~~~~~~~

::

   pip install "pyradiomics>=3.0.1,<3.2"
   python -c "import radiomics; print(radiomics.__version__)"

Or from conda-forge if you prefer::

   conda install -c conda-forge pyradiomics


Optional extras
---------------

Install only what your workflow needs. Missing extras raise
``OptionalDependencyError`` with the install hint.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Extra
     - Command
     - Needed for
   * - ``radiomics``
     - ``pip install "habitat-analysis[radiomics]"``
       (empty alias; still install PyRadiomics
       separately as above)
     - Documented alias only — does **not** install
       PyRadiomics
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
     - Every optional workflow dep **except**
       PyRadiomics (install that separately)

Combine extras::

   pip install "habitat-analysis[ml,analysis]"

From a source checkout (editable)::

   pip install -e ".[ml,analysis]"


Copy-paste recipes
------------------

Minimal library (any OS)::

   conda create -n habit python=3.10 -y
   conda activate habit
   pip install -U pip
   pip install habitat-analysis
   habit --version

Radiomics + ML (Windows, Python 3.10 example)::

   conda create -n habit python=3.10 -y
   conda activate habit
   pip install -U pip
   pip install "habitat-analysis[ml,analysis]"
   pip install https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp310-cp310-win_amd64.whl
   habit --version
   python -c "import radiomics; print(radiomics.__version__)"

Radiomics + ML (macOS / Linux)::

   conda create -n habit python=3.10 -y
   conda activate habit
   pip install -U pip
   pip install "habitat-analysis[ml,analysis]"
   pip install "pyradiomics>=3.0.1,<3.2"
   habit --version
   python -c "import radiomics; print(radiomics.__version__)"

From GitHub source::

   git clone https://github.com/lichao312214129/HABIT.git
   cd HABIT
   conda create -n habit python=3.10 -y
   conda activate habit
   pip install -U pip
   pip install .
   pip install -e ".[ml,analysis]"
   # then install PyRadiomics separately (Windows wheel URL or PyPI / conda-forge)


Common problems
---------------

**``habit`` is not recognized**

Activate the environment (``conda activate habit``), then retry. Confirm with
``where habit`` (Windows) or ``which habit`` (macOS/Linux).

**``pip install habitat-analysis`` fails mentioning ``pyradiomics``**

Base HABIT does not require PyRadiomics. Install without extras first::

   pip install habitat-analysis

**``pip install pyradiomics`` fails on Windows**

Expected. Use the matching GitHub Release wheel URL from the table above
instead of bare ``pip install pyradiomics``.

**Permission / SSL / mirror errors at hospital networks**

Ask IT to allow access to ``pypi.org`` and GitHub Releases (Windows
PyRadiomics wheels).

**More FAQ**

See :doc:`../troubleshooting/faq`.


Next steps
----------

* YAML + CLI quickstart: :doc:`quickstart`
* Python API quickstart: :doc:`quickstart_python`
* Examples gallery: :doc:`../examples/index`
