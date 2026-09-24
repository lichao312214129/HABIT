Habitat Guide
=============

Start with the complete analysis, then open one stage at a time.
Copy the Python, swap ``DATA`` / ``MODALITIES`` / ``ROI``. Figures on
a page come from that page's ``plot_*`` call.

#. :doc:`/auto_examples/00_full_pipeline/plot_01_full_pipeline` —
   one two-step study, from images to a saved model.
#. :doc:`/auto_examples/01_data_in/index` — other ways to build the cohort.
#. :doc:`/auto_examples/02_stages/index` — each ``Stage`` / ``Spec``.
#. :doc:`/auto_examples/03_quantify/index` — volume, MSI, ITH, graph,
   radiomics.
#. :doc:`/auto_examples/04_designs/index` — drop ``partition`` or ``pool``.
#. :doc:`/auto_examples/05_apply/plot_04_apply_saved_model` — reuse the model.
#. :doc:`/auto_examples/06_matching/index` — habitat ids that do not match.
#. :doc:`/auto_examples/07_parallel/plot_01_backends` — the same study
   on serial and process backends.
#. :doc:`/auto_examples/08_precision/plot_01_precise_features` —
   repeatability screening.

Every example loads the official pack with
:func:`~habit.datasets.fetch_demo` (downloads once). CLI / YAML:
:doc:`/tutorial/quickstart` and :doc:`/configuration/index`.
