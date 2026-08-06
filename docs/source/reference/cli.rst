Command reference
=================

HABIT v1.0 provides 17 subcommands. Every one is a thin shell over the
Python API (:doc:`../api/index`): a command reads its YAML, builds the same
spec objects, and calls the same recipes.

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Command
     - Purpose
     - Guide
   * - ``habit preprocess``
     - Image preprocessing
     - :doc:`../how_to/preprocess`
   * - ``habit sort-dicom``
     - DICOM sort / rename
     - :doc:`../how_to/preprocess` · :doc:`../configuration/dicom_sort`
   * - ``habit get-habitat``
     - Habitat segmentation
     - :doc:`../how_to/segment_habitat`
   * - ``habit extract``
     - Feature extraction
     - :doc:`../how_to/extract_features`
   * - ``habit radiomics``
     - Traditional radiomics
     - :doc:`../how_to/radiomics` · :doc:`../configuration/radiomics`
   * - ``habit model`` / ``habit cv``
     - ML / K-fold
     - :doc:`../how_to/train_model`
   * - ``habit compare``
     - Model comparison
     - :doc:`../how_to/compare_models` · :doc:`../configuration/model_comparison`
   * - ``habit check-config``
     - Validate a YAML without running
     - :doc:`../configuration/index`
   * - ``habit migrate-config``
     - Upgrade a v0.1 YAML to the v1 layout
     - :doc:`../api/spec` · :doc:`../configuration/index`
   * - ``habit list-components``
     - List registered components per domain (``--domain``, ``--json``)
     - :doc:`../api/plugins` · :doc:`../api/domain`
   * - ``habit icc`` / ``habit retest``
     - Reproducibility analysis
     - :doc:`../how_to/auxiliary_tools` · :doc:`auxiliary`
   * - ``habit dicom-info`` / ``merge-csv`` / ``dice``
     - Utilities
     - :doc:`../how_to/auxiliary_tools` · :doc:`auxiliary`
   * - ``habit gui``
     - Web GUI (under development)
     - :doc:`../gui/index`

Global flags: ``--help`` , ``--version`` . ``--config`` / ``-c`` applies to pipeline commands that take a YAML (not ``dice``, ``dicom-info``, ``merge-csv``, ``gui``). ``--debug`` and ``--resume`` apply to ``habit get-habitat`` only.

Templates: ``config/`` · fields: :doc:`../configuration/index` .

Errors, exit codes, and validation
----------------------------------

CLI user errors exit with code **1**. Success is **0**. There is no
fine-grained exit-code matrix in v1.0 — distinguish failure classes from the
message text (and from ``processing.log`` under ``out_dir``).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Situation
     - Expected CLI behaviour
   * - Bad YAML syntax (tabs, parse error)
     - Bilingual actionable message; **no** Python traceback
   * - Schema / required-field errors
     - ``habit check-config -c … -w <workflow>`` lists field paths
   * - Missing ``data_dir`` / empty cohort
     - ``ConfigurationError`` / ``DataFormatError`` surfaced as a short message
   * - Optional dependency missing (e.g. GUI, radiomics)
     - ``OptionalDependencyError`` with install hint
   * - Habitat subject failures under YAML ``on_subject_failure: continue``
     - Recipes proceed with successful subjects; failures are recorded on the
       run manifest. The job exits non-zero only when every subject failed
       (or ``fail_fast`` aborted) — see :doc:`../configuration/habitat` and
       :doc:`../api/execution`

Validate without running a job::

   habit check-config -c config/habitat/config_habitat_two_step.yaml -w habitat

Python-side exception matrix: :doc:`../api/exceptions`.
Runnable resilience demo: :doc:`../examples/fault_tolerance`.
