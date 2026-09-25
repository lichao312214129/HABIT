Habitat Guide
=============

**Background.** A habitat is a sub-region inside the tumour (the ROI)
whose voxels behave alike across the input images. Habitat analysis
paints every ROI voxel with a habitat id, then turns the habitat map
into numbers (volume, spatial mixing, heterogeneity, radiomics) that can
enter statistics or a model.

**Purpose.** Start with the complete analysis, then open one stage at a
time. Copy the Python, swap ``DATA`` / ``MODALITIES`` / ``ROI``. Figures
on a page come from that page's ``plot_*`` call.

* :doc:`0. Complete analysis </auto_examples/00_full_pipeline/index>` —
  one two-step study, from images to habitat maps, a feature table and a
  saved model.
* :doc:`1. Data In </auto_examples/01_data_in/index>` — build the cohort
  from a folder, image files, SimpleITK images or NumPy arrays.
* :doc:`2. Each stage </auto_examples/02_stages/index>` — what extract,
  preprocess, partition, fit and assign each do, one page per option.
* :doc:`3. Habitat Quantification </auto_examples/03_quantify/index>` —
  turn a habitat map into volume, MSI, ITH, graph and radiomics features.
* :doc:`4. Three habitat designs </auto_examples/04_designs/index>` —
  two-step, clustering inside each subject, or pooling voxels directly.
* :doc:`5. Apply a saved model </auto_examples/05_apply/index>` — label
  new patients with a saved ``.habitatmodel`` without fitting again.
* :doc:`6. Matching Habitat Labels </auto_examples/06_matching/index>` —
  make habitat ids comparable when they come from separate fits.
* :doc:`7. Running a whole cohort </auto_examples/07_parallel/index>` —
  backends, failed subjects, checkpoint resume, timeouts, and when
  parallel is worth it.
* :doc:`8. Precise Feature Screening </auto_examples/08_precision/index>` —
  keep only features that stay repeatable under simulated retest.

Every example loads the official pack with
:func:`~habit.datasets.fetch_demo` (downloads once). CLI / YAML:
:doc:`/tutorial/quickstart` and :doc:`/configuration/index`.
