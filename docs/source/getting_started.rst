Getting Started
===============

Installation
------------

Install HABIT using pip:

.. code-block:: bash

    pip install habit

Or from source:

.. code-block:: bash

    git clone https://github.com/lichao312214129/HABIT.git
    cd HABIT
    pip install -e .

Quick Start
-----------

1. Prepare your configuration file:

.. code-block:: yaml

    data_dir: /path/to/images
    out_dir: /path/to/output

2. Run habitat analysis:

.. code-block:: bash

    habit habitat --config config.yaml

Basic Usage
-----------

.. code-block:: python

    from habit.core.habitat_analysis import HabitatAnalysis

    # Load configuration
    config = {
        'data_dir': '/path/to/images',
        'out_dir': '/path/to/output',
    }

    # Create and run analysis
    analysis = HabitatAnalysis(config=config)
    results = analysis.run()

Next Steps
----------

* Read the :doc:`../user_guide/configuration` guide for configuration options
* Check :doc:`../tutorials/basic_habitat_analysis` for a step-by-step tutorial
* Explore the :doc:`../api/modules` for Python API reference
