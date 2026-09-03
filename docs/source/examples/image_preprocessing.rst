Batch image preprocessing
=========================

**Level:** recipe · **Data:** ``demo_data/preprocessed`` · **Extras:** none ·
**Time:** ~1–4 min

.. note::

   Supporting example, not the habitat core.
   :doc:`../tutorial/habitat_analysis` · :doc:`habitat_atomic_ops`.

:func:`~habit.recipes.preprocess_images` is the programmatic twin of
``habit preprocess``. The full literature-aligned pipeline on real MRI is
**N4 → registration → resample → z-score** (see
``config/preprocessing/config_preprocessing_n4_reg_resample_zscore.yaml``);
registration requires ``pip install habit[registration]`` (ANTs).

Entry points
------------

* **Batch (directory pipeline)** — ``preprocess_images(config)`` scans
  ``data_dir`` and writes ``processed_images/`` under ``out_dir``.
* **Atomic (in-memory)** — :func:`~habit.api.preprocessing.preprocess_subject` on one
  :class:`~habit.contracts.Subject`, or :func:`~habit.api.preprocessing.preprocess_image`
  on a single :class:`~habit.api.image.ImageVolume`.

This example runs **resample + z-score** on ``demo_data/preprocessed``
(LAP) and an atomic z-score of ``subj001``.

Script
------

Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed tree.

.. literalinclude:: scripts/image_preprocessing_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

Draw the figures
----------------

Paste this after the Script block (it uses ``subject``, ``processed``, and
``modality``). Writes ``out/image_preprocess_slice.png``.

.. literalinclude:: scripts/image_preprocessing_demo.py
   :language: python
   :start-after: # BEGIN figures
   :end-before: # END figures

Output (abbreviated)
--------------------

::

   === Batch: demo_data (resample + z-score) ===
     output: .../habit_preprocess_demo_...
     image files: ...

   === Atomic: z-score one subject ===
     subj001 LAP: shape=(200, 360, 360)
   Wrote out/image_preprocess_slice.png

Figures
-------

Atomic z-score of ``subj001`` LAP (original | processed, whole FOV,
greyscale — not an ROI crop):

.. figure:: ../_static/images/examples/image_preprocess_slice.png
   :alt: Original LAP beside whole-volume z-scored LAP
   :width: 720

   :func:`~habit.viz.plot_intensity_slice` after a same-grid z-score.
   Mean/std may be estimated inside the ROI (``only_inmask=True``); the
   displayed volume is still the full slice. Independent colorbars show
   raw intensity versus z-score (do not share ``vmin``/``vmax``).

What to read next
-----------------

* :doc:`../configuration/preprocessing` — every preprocessing module
* :doc:`cohort_plugins_auxiliary` — ``cohort_from_directory`` on processed data
* :doc:`habitat_preprocessing` — subject-level habitat feature chains
