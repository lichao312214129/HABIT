API Reference
=============

This is the class and function reference of HABIT.
Usage lives in the :doc:`Habitat Guide <../auto_examples/index>` and
:doc:`python_api`. Spec / component names:
:doc:`../how_to/habitat_components`.

.. note::

   Packages below are listed as links with a short task gloss.
   This page does **not** dump autosummary tables of every classifier,
   selector, or kernel. Click a package to see its classes and functions.

The declarative registry of the stable surface lives in
``habit/_public_api.py`` (``PUBLIC_NAMESPACES``). Package pages document
that export surface (each package ``__all__``). Anything not exported
there is internal and may change without notice.

Habitat analysis pipeline
-------------------------

.. toctree::
   :maxdepth: 1

   recipes
   spec
   contracts
   voxel_features
   supervoxel
   habitat_model
   habitat_features
   feature_preprocessing
   precision
   pipeline
   combiners
   report
   viz
   execution
   adapters
   datasets
   kernels
   registry
   plugins
   exceptions

Supporting bookmarks
--------------------

Tabular ML after a habitat feature table, plus image I/O / preprocessing
for data ingestion. Frozen supporting paths — not a wall of classifiers
on this index.

.. toctree::
   :maxdepth: 1

   table_ml
   image_io
   image_preprocessing

Narrative guides
----------------

Walkthroughs next to this reference (not class tables):

.. toctree::
   :maxdepth: 1

   python_api
   data_model
   domain
   domain_habitat
   domain_table
