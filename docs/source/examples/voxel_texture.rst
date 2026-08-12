Voxel texture maps (demo_data)
==============================

**Level:** atomic · **Data:** ``demo_data`` · **Extras:** ``[viz]`` ·
**Time:** ~10–40 s

This demo loads real **subj001** LAP anatomy + ROI mask from local
``demo_data/`` (not shipped in git), computes **local entropy** (fast built-in
texture; not PyRadiomics), and regenerates the gallery PNGs. The same plotter
accepts densified ``voxel_radiomics`` / custom ``VoxelFeatureField`` columns.

**Default inputs** (first existing path wins):

* Image: ``demo_data/preprocessed/images/subj001/LAP/...Series0009.nrrd``
* Mask: ``demo_data/preprocessed/masks/subj001/LAP/..._mask.nrrd``

If ``demo_data`` is missing, the script exits with a clear error. Committed
PNGs below were generated on the maintainer machine so readers still see
real-data figures without a local copy.

Script
------

.. literalinclude:: scripts/voxel_texture_demo.py
   :language: python

Run from the repository root (one line; regenerates the gallery PNGs below)::

   python docs/source/examples/scripts/voxel_texture_demo.py

What it shows
-------------

1. **Kernel path** — :func:`~habit.local_entropy_map` (arrays in, dense map out).
2. **Domain path** —
   :meth:`~habit.domain.VoxelFeatureExtractorRegistry.create`\ ``("local_entropy", ...)``
   → :func:`~habit.viz.dense_voxel_feature_map`.
3. **Publication figures** — :func:`~habit.viz.plot_voxel_texture_slice` with
   :func:`~habit.viz.use_style` (``radiology`` / ``nature``). Volumes are
   cropped to the padded ROI bbox before rendering.

For the how-to narrative see :doc:`../how_to/voxel_texture`. For habitat
**graph** topology figures (nodes/edges on habitat maps) see
:doc:`graph_features`.

Publication figures
-------------------

The demo writes English-labelled PNGs under
``docs/source/_static/images/examples/``.

Side-by-side (anatomy | texture)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_static/images/examples/voxel_texture_side_by_side.png
   :alt: Demo subj001 LAP anatomy next to local-entropy map
   :width: 720

   Densest axial slice: greyscale LAP and local entropy inside the ROI
   (:func:`~habit.viz.plot_voxel_texture_slice`, ``mode="side_by_side"``).

Overlay on anatomy
~~~~~~~~~~~~~~~~~~

.. figure:: ../_static/images/examples/voxel_texture_overlay.png
   :alt: Local entropy translucent overlay on LAP anatomy
   :width: 480

   Same slice with ``mode="overlay"`` and the ``nature`` style preset.

Orthogonal triptych
~~~~~~~~~~~~~~~~~~~

.. figure:: ../_static/images/examples/voxel_texture_orthogonal.png
   :alt: Orthogonal local-entropy panels for demo subj001
   :width: 520

   Three orthogonal planes through the densest ROI support (default 3D
   layout when ``axis`` is omitted).

Also see
--------

* How-to: :doc:`../how_to/voxel_texture`
* API: :func:`~habit.viz.plot_voxel_texture_slice`,
  :func:`~habit.viz.dense_voxel_feature_map`,
  :func:`~habit.kernels.local_entropy_map`
