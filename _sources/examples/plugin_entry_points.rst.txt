Third-party plugins (entry points)
==================================

**Level:** extending HABIT · **Data:** none · **Extras:** none · **Time:** <5 s

Two registration routes:

1. **In-process** — ``@VoxelFeatureExtractorRegistry.register("name")`` inside
   the running interpreter (see :doc:`custom_voxel_features`).
2. **Entry point** — a separate package declares
   ``habit.<domain>`` in ``pyproject.toml``, users call
   :func:`~habit.load_plugins` once, then reference the name in
   ``Spec("name", {...})``.

Domain string = snake_case singular protocol name
(e.g. ``voxel_feature_extractor``). Group name = ``habit.`` + domain.

Minimal ``pyproject.toml`` fragment::

   [project.entry-points."habit.voxel_feature_extractor"]
   t1_t2_contrast = "my_pkg.features:register"

``register()`` should perform the decorator registration (or
``Registry.register_factory``) when imported.

Script
------

.. literalinclude:: scripts/plugin_entry_points_demo.py
   :language: python

Run::

   python docs/source/examples/scripts/plugin_entry_points_demo.py

What to read next
-----------------

* :doc:`custom_voxel_features` — full DIY extractor + habitat run
* :doc:`habitat_custom_pipeline` — plug the name into Spec stages
* :doc:`../customization/index` — all extensible domains
* :doc:`../api/plugins` — ``list_plugins`` / ``get_param_schema``
