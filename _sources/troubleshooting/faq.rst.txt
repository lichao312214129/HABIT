FAQ
===

Installation
------------

**``habit`` not found**

- Activate your environment (``conda activate habit`` or the venv you used),
  then ``habit --version``.
- Full step-by-step: :doc:`../tutorial/installation`.

**``Parquet export requires pyarrow``**

Current default installs include ``pyarrow``. Reinstall / upgrade
``habitat-analysis`` in the same environment. Temporary workaround:
``pip install "pyarrow>=15,<22"`` (or ``pyarrow>=22,<23`` on Python 3.14).

**``pip install habitat-analysis`` / ``pip install pyradiomics`` fails**

Follow :doc:`../tutorial/installation`. Short version:

- Base ``pip install habitat-analysis`` does **not** need PyRadiomics.
- Install PyRadiomics **separately** (HABIT extras never pull it).
- PyPI has **no usable PyRadiomics Windows binaries**: bare
  ``pip install pyradiomics`` pulls the broken **3.1.0 sdist**. On Windows
  install the matching prebuilt wheel from Release ``v1.0.2``, for example
  (Python 3.10)::

     pip install https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp310-cp310-win_amd64.whl

  On macOS / Linux: ``pip install "pyradiomics>=3.0.1,<3.2"`` or
  ``conda install -c conda-forge pyradiomics``.
- habitat-analysis ≥ 1.0.4 supports Python 3.10–3.14 with numpy 1.26 or 2.x.

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
