Legacy compat (``habit.compat``)
=================================

v2 removed the third-party interop adapters that used to live under
``habit.compat.monai``, ``habit.compat.nnunet``, and
``habit.compat.sklearn``. Those modules are gone and must not be
imported. Do not expect Sphinx autodoc pages for them.

Core tabular ML still uses :class:`~habit.pipeline.TablePipeline` (a
scikit-learn ``Pipeline`` subclass) and
:mod:`habit.pipeline.sklearn_interop` internally; those are not
third-party interop surfaces.

``habit.compat`` itself exports nothing (``__all__ = []``). Remaining
submodules are frozen v0.1 orchestration helpers pending further
migration or deletion:

* ``habit.compat.engines`` — legacy habitat / ML / preprocessing runners
* ``habit.compat.dicom_sort_runner``
* ``habit.compat.plugin_registries``
* ``habit.compat.registry_facade``
* ``habit.compat.legacy_core``

These are not part of the v2 public API. Prefer
:mod:`habit.recipes`, :mod:`habit.pipeline`, and
:func:`habit.precision.align_habitat_map`.
