Graph topology features (synthetic)
===================================

**Level:** atomic · **Data:** synthetic · **Extras:** ``[viz]`` / ``[view]`` for
figures · **Time:** ~10–30 s (3D off-screen)

``graph`` is a **built-in** habitat feature family. This demo uses synthetic
labels only (no ``demo_data/``). Prefer the domain / kernel API shown here;
``habit.compat.graph_plugin`` is a deprecated transitional shim.

Script
------

.. literalinclude:: scripts/graph_features_demo.py
   :language: python

Run from the repository root (one line; regenerates the gallery PNGs below)::

   python docs/source/examples/scripts/graph_features_demo.py

What it shows
-------------

1. **Kernel path** — :func:`~habit.extract_graph_features` with
   :class:`~habit.HabitatGraphFeatureOptions` (arrays in, ``dict`` out).
2. **Domain path** —
   :meth:`~habit.domain.HabitatFeatureExtractorRegistry.create`\ ``("graph", ...)``
   on a :class:`~habit.contracts.Subject` + :class:`~habit.contracts.HabitatMap`.
3. **Publication figures** — :mod:`habit.viz.habitat_graph` with
   :func:`~habit.viz.use_style` (2D matplotlib; 3D PyVista off-screen).

For CLI / YAML (``feature_types: [graph]`` + optional ``graph:`` block) see
:doc:`../how_to/extract_features` and
:doc:`../configuration/feature_extraction`. Column definitions:
:doc:`../reference/features/graph`.

Publication figures
-------------------

The demo writes English-labelled PNGs under
``docs/source/_static/images/examples/``. 2D panels use the
``radiology`` / ``nature`` style presets; 3D scenes use a white academic
background (requires ``pyvista`` and ``scikit-image``).

2D habitat slice
~~~~~~~~~~~~~~~~

.. figure:: ../_static/images/examples/graph_habitat_slice_2d.png
   :alt: Synthetic habitat label map at the largest cross-section
   :width: 480

   Habitat labels on the representative cross-section
   (:func:`~habit.viz.plot_habitat_graph_slice`).

2D region network
~~~~~~~~~~~~~~~~~

.. figure:: ../_static/images/examples/graph_habitat_network_2d.png
   :alt: Intra- and inter-habitat graphs on a 2D habitat slice
   :width: 720

   Intra-habitat panels plus the combined inter-habitat graph
   (:func:`~habit.viz.plot_habitat_graph_network_2d`).

3D habitat surfaces
~~~~~~~~~~~~~~~~~~~

.. figure:: ../_static/images/examples/graph_habitat_surface_3d.png
   :alt: Off-screen PyVista surface render of three synthetic habitats
   :width: 520

   Marching-cubes habitat surfaces
   (:func:`~habit.viz.render_habitat_graph_surface_3d`).

3D spatial network
~~~~~~~~~~~~~~~~~~

.. figure:: ../_static/images/examples/graph_habitat_network_3d.png
   :alt: Off-screen PyVista 3D graph with habitat-colored nodes and edges
   :width: 520

   Centroid nodes with intra- and inter-habitat edges
   (:func:`~habit.viz.render_habitat_graph_network_3d`).

The extract recipe can also write
``<out_dir>/visualizations/graph/`` when ``graph.visualize: true``.

What to read next
-----------------

* :doc:`feature_extraction` — cohort extract recipe after habitat maps exist
* :doc:`visualization` — other ``habit.viz`` figures
* :doc:`../api/domain_habitat` — registry pattern for all habitat families
