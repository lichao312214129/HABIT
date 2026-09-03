HABIT Documentation
=====================

**HABIT** turns images + an ROI into habitat maps, then quantifies those
habitats. Supporting image preprocessing is provided for data ingestion.

.. warning::

   Research / education only. Not for clinical diagnosis.

Start here
----------

1. :doc:`tutorial/installation` — install
2. :doc:`auto_quickstart/plot_quickstart_python` — first habitat map (Python)
3. :doc:`tutorial/quickstart` — first habitat map (CLI / YAML)
4. :doc:`auto_examples/index` — **Habitat Guide** (one task per page)

Stuck: :doc:`troubleshooting/faq`.
Signatures and defaults: :doc:`api/index`.
Scientific definitions: :doc:`reference/features/index`.


.. toctree::
   :maxdepth: 1
   :caption: Get started

   tutorial/installation
   auto_quickstart/plot_quickstart_python
   tutorial/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Habitat Guide
   :titlesonly:

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   how_to/habitat_components
   reference/features/index
   configuration/index
   reference/cli
   troubleshooting/faq
   reference/upstream_libraries

.. toctree::
   :maxdepth: 1
   :caption: Developer

   development/architecture
   development/contributing
   customization/index

.. toctree::
   :maxdepth: 1
   :caption: Changelog

   changelog

.. Do not add another toctree after Acknowledgments. Sphinx / RTD parents
   a later toctree onto the last caption, which would dump leftover pages
   into the Acknowledgments sidebar. CLI / YAML how_to bookmark pages stay
   :orphan: (reachable by URL / cross-links). Guide duplicates stay :orphan:.

.. toctree::
   :maxdepth: 1
   :caption: Acknowledgments

   acknowledgments

- GitHub: |github_repo|
