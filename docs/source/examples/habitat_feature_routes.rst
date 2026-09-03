Voxel features
==============

Clustering uses the voxel field you define — not a fixed T1 image.
This page is the intensity pair: ``raw`` (concatenate modality intensities
inside the ROI) and ``concat`` (join families column-wise, e.g.
``concat(raw("T1"), raw("T2"))``). Custom formulas and plugins:
:doc:`custom_voxel_features`. Texture maps: :doc:`voxel_texture`.

.. literalinclude:: scripts/habitat_feature_routes_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

.. literalinclude:: scripts/habitat_feature_routes_demo.py
   :language: python
   :start-after: # BEGIN figures
   :end-before: # END figures

.. figure:: ../_static/images/examples/habitat_feature_routes_overlay.png
   :alt: Habitat overlay after a raw-intensity feature route
   :width: 420

   Habitats after ``Study(...).fit_predict`` with ``raw`` intensities
   (:func:`~habit.viz.plot_habitat_overlay`).

**Next:** :doc:`custom_voxel_features` · :doc:`voxel_texture`
