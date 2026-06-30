FAQ
===

Installation
------------

**``habit`` not found**

- Portable: run ``setup_habit.bat`` and open a **new** terminal.
- Source: ``conda activate habit`` .

**No ``python.exe`` after extract**

Extract one level too deep — move files up to the target folder.

**CUDA False**

Normal on CPU pack. GPU pack: check NVIDIA drivers; see :doc:`../tutorial/installation` .

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

GUI
---

**Browser does not open** — try ``habit gui --port 8502`` ; see :doc:`../gui/index` .

Data
----

**ROI format**: NIfTI aligned with images (ITK-SNAP / 3D Slicer).

**Folder layout**: :doc:`../how_to/prepare_data` .

Support: `GitHub Issues <|github_issues|>`_ · lichao19870617@163.com
