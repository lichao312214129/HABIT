CLI and YAML
============

YAML templates and ``habit`` commands live only here, not in the Habitat
Guide. Copy a file from ``config/``, edit the ★ fields (usually data and
output paths), and run the matching command.

Command list: :doc:`../reference/cli`.
Catalog of templates: :doc:`recipe_catalog`.

.. toctree::
   :maxdepth: 2

   recipe_catalog
   habitat
   feature_extraction
   radiomics
   preprocessing
   dicom_sort
   auxiliary

Supporting bookmarks
--------------------

These are not Habitat Guide pages. Templates live under ``config/``.

* Image preprocessing (N4, resample, registration): :doc:`preprocessing`
* DICOM sort / rename: :doc:`dicom_sort`
* Dice, ICC, merge-csv, and other utilities: :doc:`auxiliary`
* Whole-ROI radiomics CLI: :doc:`radiomics`
