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

- On **macOS / Linux**, ``conda install -c conda-forge pyradiomics`` is fine
  when Conda can reach conda-forge (or a working mirror). On **Windows** do
  not treat Conda as the primary path: mirror / ``repodata`` timeouts and
  unsatisfiable solves are common; use the Release wheel above instead.
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
4. Validate YAML first: ``habit check-config -c <yaml> -w <workflow>``
   (exit code 1, no traceback on user errors — see :doc:`../reference/cli`).

**Subject failed but others finished**

Habitat YAML ``on_subject_failure: continue`` (default) isolates per-subject
errors; recipes / CLI proceed with successful subjects and record exclusions
on the run manifest. In the Python API, default ``cohort.map`` still raises
``ProcessingError``; pass ``raise_on_failure=False`` (or call ``backend.map``)
for soft failure — see :doc:`../examples/fault_tolerance` and
:doc:`../api/execution`.

**Timeout / OOM / parallel_mode seem ignored**

On the v1 CLI / ``run_from_yaml`` path those knobs apply when
``ProcessPoolBackend`` is selected (``backend: process`` **or** a positive
per-subject timeout, including ``processes: 1``). True in-process serial
requires ``individual_subject_timeout_sec: null``. See :doc:`../api/execution`
and the mapping table in :doc:`../api/spec`.

**strict_checkpoint_hash raises on mismatch**

With ``strict_checkpoint_hash: true``, an incompatible ``run_fingerprint.json``
raises :class:`~habit.exceptions.CompatibilityError`. A readable v0.1
``manifest.json`` / ``subjects/`` checkpoint is auto-migrated then resumed;
only a corrupt/unreadable legacy manifest raises. Details:
:doc:`../configuration/habitat`.

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
