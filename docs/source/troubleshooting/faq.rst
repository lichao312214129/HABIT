FAQ
===

Install
-------

**``habit`` not found** — you are not in the conda env. On Windows open
**Anaconda Prompt** (not plain CMD), run ``conda activate habit`` until
the prompt shows ``(habit)``, then ``habit --version``.
See :doc:`../tutorial/installation`.

**``pip install`` fails / mirror missing package** — use official PyPI::

   pip install habitat-analysis -i https://pypi.org/simple

**PyRadiomics on Windows** — do **not** use bare ``pip install pyradiomics``
(broken sdist). Install the wheel from Release ``v1.0.2`` matching your
Python, e.g. 3.10::

   pip install https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp310-cp310-win_amd64.whl

**napari / ``PluginManifest`` / pydantic errors**::

   pip install "pydantic>=2.12,<3"
   pip install "napari[pyqt5]" "npe2>=0.8.2" "pydantic!=2.11.*,>=2.8,<3" -i https://pypi.org/simple

**napari window flashes then closes** — keep ``show=True`` (default); the
call blocks until you close the window.

**``OptionalDependencyError``** — the message includes the
``pip install ...`` command for the missing extra.

**Parquet / pyarrow** — either ``pip install "habitat-analysis[tables]"``
or set ``habitats_results_format: csv`` in YAML.

Run
---

**Path / file errors** — run from the project root (folder that contains
``config/``). Check ``data_dir`` / ``out_dir`` (and absolute paths if unsure).

**Command failed** — ``habit <cmd> --help``; read ``processing.log`` in the
output folder; validate with ``habit check-config -c <yaml>``.

**YAML changes ignored** — confirm the file passed with ``-c``; CLI
``--mode`` overrides YAML ``run_mode``.

Data
----

ROI: NIfTI aligned with images. Layout / path-list YAML:
:doc:`../how_to/prepare_data`.

Support: |link_github_issues| · lichao19870617@163.com
