Graph topology features (synthetic)
===================================

**Level:** atomic · **Data:** synthetic · **Extras:** none · **Time:** ~1 s

``graph`` is a **built-in** habitat feature family. This demo uses synthetic
labels only (no ``demo_data/``). Prefer the domain / kernel API shown here;
``habit.compat.graph_plugin`` is a deprecated transitional shim.

Script
------

.. literalinclude:: scripts/graph_features_demo.py
   :language: python

Run::

   python docs/source/examples/scripts/graph_features_demo.py

What it shows
-------------

1. **Kernel path** — :func:`~habit.extract_graph_features` with
   :class:`~habit.HabitatGraphFeatureOptions` (arrays in, ``dict`` out).
2. **Domain path** —
   :meth:`~habit.domain.HabitatFeatureExtractorRegistry.create`\ ``("graph", ...)``
   on a :class:`~habit.contracts.Subject` + :class:`~habit.contracts.HabitatMap`.

For CLI / YAML (``feature_types: [graph]`` + optional ``graph:`` block) see
:doc:`../how_to/extract_features` and
:doc:`../configuration/feature_extraction`. Column definitions:
:doc:`../reference/features/graph`.

Optional figures
----------------

Pure plotters live under :mod:`habit.viz` (``[viz]``; 3D needs ``[view]``)::

   from habit.viz import plot_habitat_graph_slice, plot_habitat_graph_network_2d

   fig = plot_habitat_graph_slice(labels)
   fig2 = plot_habitat_graph_network_2d(labels, options=options)

The extract recipe can also write
``<out_dir>/visualizations/graph/`` when ``graph.visualize: true``.

What to read next
-----------------

* :doc:`feature_extraction` — cohort extract recipe after habitat maps exist
* :doc:`visualization` — other ``habit.viz`` figures
* :doc:`../api/domain_habitat` — registry pattern for all habitat families
