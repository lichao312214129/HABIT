Image preprocessing (bookmark)
==============================

Frozen supporting path for **data ingestion** (resample, N4, registration).
HABIT's product core is habitat analysis — this page is a bookmark, not a
catalog of preprocessor classes.

**User guide:** :doc:`python_api`. Prefer
:func:`~habit.recipes.preprocess_images` for a cohort, or
:func:`~habit.recipes.preprocess_subject` / :func:`~habit.recipes.preprocess_image`
for one volume.

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   habit.recipes.preprocess_images
   habit.recipes.preprocess_subject
   habit.recipes.preprocess_image
