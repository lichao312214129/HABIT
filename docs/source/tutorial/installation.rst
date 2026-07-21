Installation
============

**Windows (recommended)**: Download the lightweight ZIP → extract → double-click
``一键安装HABIT.bat``. The installer creates an isolated Python 3.10 environment
inside the extracted directory; users do not need to install Python or Conda.

**macOS / Linux / developers**: See **Source install** below.

Demo data: :doc:`quickstart` .

Windows lightweight installer
-----------------------------

1. Extract the ZIP to a short local path containing only ASCII characters and
   no spaces, for example ``D:\HABIT``.
2. Double-click ``一键安装HABIT.bat`` and wait for the environment and capability
   checks to finish.
3. Double-click ``启动HABIT命令行.bat`` to open a terminal configured only for
   this HABIT installation.
4. Run ``一键启用HABIT-AutoML.bat`` only for AutoGluon Tabular workflows, or
   ``一键启用HABIT-进阶分析.bat`` only for Parquet, SHAP, Plotly, Krippendorff,
   and survival-analysis features.
5. On a compatible NVIDIA system, optionally run ``一键启用HABIT-GPU.bat`` after
   the CPU installation succeeds. A failed GPU enhancement does not invalidate
   the CPU environment.

The first installation requires network access. The installer validates the
installation path, disk space, package assets, and the pinned Python 3.10
runtime before it changes the project-managed ``.mamba`` directory. The default
profile deliberately excludes AutoML, advanced analysis, and Torch; optional
entry points add only the selected feature family.

Source install
--------------

.. code-block:: bash

   conda create -n habit python=3.10
   conda activate habit
   cd /path/to/habit_project_v1
   pip install -e .
   habit --version

Install an optional feature family only when needed: ``pip install -e ".[analysis]"``,
``pip install -e ".[automl]"``, or ``pip install -e ".[torch]"``. Windows may
also need ``pip install installer/vendor/pyradiomics-3.0.1-cp310-cp310-win_amd64.whl`` .
Troubleshooting: :doc:`../troubleshooting/faq` .

Next → :doc:`quickstart`
