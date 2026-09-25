CLI and YAML
============

The ``habit`` command line and the YAML files are a shell over the Python
API: every command runs the same analysis as the Python pages. YAML
templates and ``habit`` commands live only in this section, not in the
Habitat Guide.

Two subpages:

* :doc:`../reference/cli` -- every ``habit`` subcommand, what it does,
  and where its options are explained.
* :doc:`recipe_catalog` -- the YAML templates under ``config/`` and the
  field reference for each workflow. Copy a file, edit the ★ fields
  (usually data and output paths), and run the matching command.

.. toctree::
   :maxdepth: 2

   ../reference/cli
   recipe_catalog

Supporting bookmarks
--------------------

These are not Habitat Guide pages. Templates live under ``config/``.

* Image preprocessing (N4, resample, registration): :doc:`preprocessing`
* DICOM sort / rename: :doc:`dicom_sort`
* Dice, ICC, merge-csv, and other utilities: :doc:`auxiliary`
* Whole-ROI radiomics CLI: :doc:`radiomics`
