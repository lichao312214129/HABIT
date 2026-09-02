HABIT Documentation
=====================

**HABIT** — habitat maps from images, then quantify those habitats.

Image preprocessing and tabular ML are **supporting** tools. The product
is the habitat API: ``op(subject)``, :class:`~habit.domain.SubjectPipeline`,
or :class:`~habit.recipes.Study`.

.. warning::

   Research / education only. Not for clinical diagnosis.

Start here
----------

1. :doc:`tutorial/installation` — install
2. :doc:`tutorial/quickstart` — **run the demo** (YAML + CLI)
3. :doc:`tutorial/habitat_analysis` — what habitats are, which strategy,
   how to embed operators
4. :doc:`how_to/prepare_data` — point HABIT at **your** images

Python beginners: :doc:`tutorial/quickstart_python` (copy-paste
:class:`~habit.recipes.Study`). Embed one step in your own workflow:
:doc:`examples/habitat_atomic_ops`. Parallel / fault tolerance:
:doc:`tutorial/execution`. Stuck: :doc:`troubleshooting/faq`.

Also: :doc:`how_to/graph_features` (topology) ·
:doc:`how_to/voxel_texture` (texture maps as inputs).

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   tutorial/installation
   tutorial/quickstart
   tutorial/quickstart_python
   tutorial/habitat_analysis
   tutorial/execution

.. toctree::
   :maxdepth: 1
   :caption: How-to guides

   how_to/index

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Configuration

   configuration/index

.. toctree::
   :maxdepth: 2
   :caption: Features from habitat maps

   reference/features/index

.. toctree::
   :maxdepth: 1
   :caption: Command reference

   reference/cli
   reference/auxiliary

.. toctree::
   :maxdepth: 1
   :caption: FAQ

   troubleshooting/faq

.. toctree::
   :maxdepth: 2
   :caption: Python API

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/index
   customization/index
   reference/upstream_libraries

.. toctree::
   :maxdepth: 1
   :caption: Changelog

   changelog

.. toctree::
   :maxdepth: 1
   :caption: Acknowledgments

   acknowledgments

- GitHub: |github_repo|
