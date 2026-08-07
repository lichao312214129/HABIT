DICOM Sort Configuration
========================

Standalone DICOM rename / reorganize via ``dcm2niix`` (``habit sort-dicom``).
This is **not** the full preprocessing pipeline; for resample / N4 / registration
see :doc:`preprocessing`.

Schema: ``DicomSortConfig``. Template:
``config/dicom_sort/config_sort_dicom.yaml``.

API reference: :doc:`../api/dicom_sort`.

**Example configuration:**

.. code-block:: yaml

   data_dir: ../../demo_data/dicom
   out_dir: ../../demo_data/results/sorted_dicom
   dcm2niix_path: ../../tools/bin/dcm2niix.exe
   f: "subj_%n_%g_%x/%s_%d/%r_%o.dcm"
   extra_args: []

Top-level fields
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Field
     - Default
     - Description
   * - ``data_dir``
     - none (required)
     - Input DICOM root; relative paths resolve against the YAML directory
   * - ``out_dir``
     - none (required)
     - Output root for sorted files
   * - ``f``
     - none (required*)
     - dcm2niix ``-f`` pattern, passed **verbatim** (no path rewriting)
   * - ``filename_format``
     - ``null``
     - Deprecated alias for ``f``
   * - ``dcm2niix_path``
     - ``null``
     - Executable file or directory; omit to search ``PATH``
   * - ``extra_args``
     - ``[]``
     - Extra CLI tokens appended to the dcm2niix command
   * - ``output_dir``
     - ``null``
     - If set, used as dcm2niix ``-o`` instead of ``out_dir``

\* Provide either ``f`` or ``filename_format``.

Notes
-----

- Do not put path-like tokens in ``f`` that HABIT would treat as file paths;
  the loader intentionally skips path resolution for the ``-f`` pattern.
- On Windows lightweight installs, ``tools/bin/dcm2niix.exe`` is the usual
  bundled binary.
- Command walkthrough: :doc:`../how_to/preprocess`.
