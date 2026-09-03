.. _api-report:

:mod:`habit.report`: run-scoped persistence and figures
=======================================================

.. automodule:: habit.report
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.report

**User guide:** Habitat Guide
:doc:`../auto_examples/05_quantify/plot_01_volume_fractions` ·
:doc:`python_api`.

Pass a :class:`~habit.report.Report` to
:meth:`~habit.recipes.Study.fit_predict`. This is **not** part of
:class:`~habit.spec.HabitatSpec` and does not enter scientific
fingerprints. ``figure_layout`` is ``"flat"`` or ``"by_subject"``
(:data:`~habit.report.FIGURE_LAYOUTS`).

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Report
   FigureAtom
   SubjectContext
   Overlay
   VolumeFractions
   MSI
   ITH
   ClusterValidation
   GraphSlice
   GraphNetwork2D

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   coerce_report
