Installation
============

Install
-------

::

   conda create -n habit python=3.10 -y
   conda activate habit
   pip install -U pip
   pip install habitat-analysis -i https://pypi.org/simple
   habit --version

* Package on PyPI: **``habitat-analysis``** (code: ``import habit``)
* Python: **3.10–3.14**
* Next: :doc:`quickstart`

If ``pip`` cannot find the package on a local mirror, keep
``-i https://pypi.org/simple``.


Optional: napari (``habit view``)
---------------------------------

::

   pip install "napari[pyqt5]" "npe2>=0.8.2" "pydantic!=2.11.*,>=2.8,<3" -i https://pypi.org/simple

From a git clone you can use ``pip install -e ".[view]"`` instead.


From source (developers)
------------------------

::

   git clone https://github.com/lichao312214129/HABIT.git
   cd HABIT
   conda create -n habit python=3.10 -y
   conda activate habit
   pip install -U pip
   pip install -e .
   habit --version


Need more later?
----------------

Missing optional packages raise ``OptionalDependencyError`` with the exact
``pip install`` command. Install only what you use::

   pip install "habitat-analysis[ml,analysis]"

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Extra
     - Needed for
   * - ``viz``
     - Figures (``habit.viz``, ML / clustering plots)
   * - ``tables``
     - ``.parquet`` / ``.xlsx`` (default habitats table is parquet)
   * - ``dicom``
     - ``habit dicom-info`` / ``habit sort-dicom``
   * - ``slic``
     - SLIC supervoxels (default ``kmeans`` / ``gmm`` need nothing)
   * - ``ml``
     - XGBoost, SMOTE, mRMR / VIF / stepwise (includes ``viz``, ``tables``)
   * - ``analysis``
     - SHAP, Plotly, ICC, survival (includes ``viz``, ``tables``)
   * - ``registration``
     - ANTs registration
   * - ``automl``
     - AutoGluon Tabular
   * - ``torch``
     - TorchRadiomics / GPU texture
   * - ``all``
     - All of the above except ``torch`` and PyRadiomics
   * - ``view``
     - napari for ``habit view`` (clone: ``pip install -e ".[view]"``)


PyRadiomics (only if you need radiomics)
----------------------------------------

Not installed by default. Needed for ``voxel_radiomics`` /
``supervoxel_radiomics``, ``habit radiomics``, and texture habitat features.

**Windows** — use a prebuilt wheel from
`Release v1.0.2 <https://github.com/lichao312214129/HABIT/releases/tag/v1.0.2>`_
(do **not** use bare ``pip install pyradiomics``)::

   # Python 3.10 example — change cp310 for 3.11–3.14
   pip install https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp310-cp310-win_amd64.whl

**macOS / Linux**::

   pip install "pyradiomics>=3.0.1,<3.2"


Problems?
---------

See :doc:`../troubleshooting/faq`. Then :doc:`quickstart` or
:doc:`quickstart_python`.
