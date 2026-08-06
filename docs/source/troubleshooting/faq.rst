FAQ
===

Installation
------------

**``habit`` not found**

- Windows lightweight installer: double-click ``launchers/启动HABIT命令行.bat``
  (after ``launchers/一键安装HABIT.bat`` succeeds).
- pip / Conda: ``conda activate habit`` then ``habit --version``.
- Full step-by-step: :doc:`../tutorial/installation`.

**Installer path rejected**

Use a short local path with ASCII characters only and no spaces
(for example ``D:\HABIT``). Avoid Desktop paths that contain Chinese
characters or spaces.

**CUDA False**

Normal for the default CPU environment. After ``launchers/一键启用HABIT-GPU.bat``,
check NVIDIA drivers; see :doc:`../tutorial/installation` .

**``Parquet export requires pyarrow``**

Current default installs include ``pyarrow``. Rebuild or re-run
``launchers/一键安装HABIT.bat`` so the environment picks up the updated runtime
lock. Temporary workaround: ``pip install pyarrow==20.0.0`` inside the HABIT
command-line session.

**``pip install habitat-analysis`` / ``pip install pyradiomics`` fails**

Follow :doc:`../tutorial/installation` (Path B + Install PyRadiomics). Short version:

- Base ``pip install habitat-analysis`` does **not** need PyRadiomics.
- PyPI has **no usable PyRadiomics Windows binaries**: bare
  ``pip install pyradiomics`` pulls the broken **3.1.0 sdist**, and 3.0.1
  has no Windows wheels either. On Windows install the prebuilt 3.1.0 wheel
  (cp310–cp314) from the HABIT GitHub Release first, then::

     pip install "habitat-analysis[radiomics]"

- Or: ``conda install -c conda-forge pyradiomics``
- habitat-analysis ≥ 1.0.2 extras allow ``pyradiomics>=3.0.1,<3.2`` and
  support Python 3.10–3.14 with numpy 1.26 or 2.x.

Runtime
-------

**Path / file errors**

- Run from project root (contains ``config/`` ).
- Check ``data_dir`` / ``out_dir`` in YAML.
- Relative paths are resolved from the YAML file directory.

**Command failed**

1. ``habit <subcommand> --help``
2. Check ``processing.log`` in the output folder.
3. ``habit get-habitat ... --debug`` for habitat jobs.

**YAML changes ignored**

Confirm you edit the file passed with ``-c`` ; ``--mode`` on CLI overrides ``run_mode`` in YAML.

GUI (under development)
-----------------------

The Web GUI is not yet recommended for routine use. Prefer the CLI and YAML
configs; see :doc:`../gui/index` for the planned scope and developer preview.

Data
----

**ROI format**: NIfTI aligned with images (ITK-SNAP / 3D Slicer).

**Folder layout**: :doc:`../how_to/prepare_data` .

Support: |link_github_issues| · lichao19870617@163.com
