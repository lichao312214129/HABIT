HABIT Documentation
=====================

**HABIT** turns medical images plus an ROI into tumour **habitat maps**,
quantifies them (volume fractions, MSI, ITH, radiomics, and HABIT's own
**habitat graph-network features**), and saves a reusable
``.habitatmodel`` so new patients can be labelled without refitting.

Project repository: https://github.com/lichao312214129/HABIT

.. warning::

   Research / education only. Not for clinical diagnosis.

.. container:: habit-home-cards

   .. container:: habit-home-card

      :doc:`Quickstart <auto_quickstart/plot_quickstart_python>`

      First habitat map in Python

   .. container:: habit-home-card

      :doc:`Examples <auto_examples/index>`

      Tutorials and how-tos

   .. container:: habit-home-card

      :doc:`User guide <user_guide/index>`

      Concepts without full code

Featured examples
-----------------

* :doc:`auto_examples/00_introductory_tutorials/plot_01_two_step` — two-step habitats
* :doc:`auto_examples/04_quantifying_habitats/plot_04_graph_network_features` — graph-network features
* :doc:`auto_examples/05_validation_and_reuse/plot_06_reuse_published_model` — reuse a published model

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   auto_quickstart/plot_quickstart_python
   auto_quickstart/plot_quickstart_yaml

.. toctree::
   :maxdepth: 1
   :caption: Installation

   tutorial/installation
   user_guide/optional_dependencies

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :titlesonly:

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index
   configuration/recipe_catalog
   reference/cli
   reference/habitat_matching

.. toctree::
   :maxdepth: 2
   :caption: Feature formulas

   reference/features/index

.. toctree::
   :maxdepth: 1
   :caption: Glossary

   user_guide/glossary

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/architecture
   customization/index

.. toctree::
   :maxdepth: 1
   :caption: Citing HABIT

   acknowledgments
