Installation
============

**Windows (recommended)**: Download the lightweight ZIP → extract → double-click
``launchers/一键安装HABIT.bat``. The installer creates an isolated Python 3.10 environment
inside the extracted directory; users do not need to install Python or Conda.

**macOS / Linux / developers**: See **Source install** below.

Demo data: :doc:`quickstart` .

Windows lightweight installer
-----------------------------

1. Extract the ZIP to a short local path containing only ASCII characters and
   no spaces, for example ``D:\HABIT``.
2. Open ``launchers/``, double-click ``一键安装HABIT.bat``, and wait for the environment and capability
   checks to finish.
3. Double-click ``launchers/启动HABIT命令行.bat`` to open a terminal configured only for
   this HABIT installation.
4. Run ``launchers/一键启用HABIT-AutoML.bat`` only for AutoGluon Tabular workflows, or
   ``launchers/一键启用HABIT-进阶分析.bat`` only for SHAP, Plotly, Krippendorff,
   and survival-analysis features. Default habitats parquet export uses ``pyarrow``
   from the base install.
5. On a compatible NVIDIA system, optionally run ``launchers/一键启用HABIT-GPU.bat`` after
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
   cd /path/to/habit_project
   pip install -e .
   habit --version

The base install is deliberately light: it covers image I/O, preprocessing,
habitat analysis, and the core table-ML path. Optional feature families are
extras — install only what a workflow needs:

.. list-table::
   :header-rows: 1

   * - Extra
     - Pulls in
     - Needed for
   * - ``.[ml]``
     - xgboost, imbalanced-learn, mrmr-selection, statsmodels
     - XGBoost model, SMOTE resampling, stepwise / univariate-logistic / VIF / mRMR selectors
   * - ``.[registration]``
     - antspyx
     - ANTs-based image registration in preprocessing
   * - ``.[analysis]``
     - shap, plotly, lifelines, scikit-survival, krippendorff, pingouin
     - SHAP explanations, interactive 3-D plots, survival analysis, ICC / rater-reliability
   * - ``.[automl]``
     - autogluon.tabular
     - AutoGluon Tabular workflows
   * - ``.[torch]``
     - torch
     - TorchRadiomics / GPU-backed feature extraction
   * - ``.[all]``
     - every family above plus the GUI server deps
     - full workstation install

For example: ``pip install -e ".[ml,analysis]"``. Using a feature whose extra
is missing raises ``OptionalDependencyError`` naming the extra to install.
Windows may also need
``pip install installer/vendor/pyradiomics-3.0.1-cp310-cp310-win_amd64.whl`` .
Troubleshooting: :doc:`../troubleshooting/faq` .

Next → :doc:`quickstart`
