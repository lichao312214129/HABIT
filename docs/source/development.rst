Development Guide
=================

Architecture
------------

HABIT follows a modular architecture:

* **Core Module**: Core functionality (habitat_analysis, machine_learning, preprocessing)
* **Common Module**: Shared utilities (configuration, logging, data processing)
* **CLI Module**: Command-line interface

Code Style
----------

HABIT uses:

* **Black** for code formatting
* **isort** for import sorting
* **pydantic** for configuration validation
* **Google-style docstrings** (via Napoleon)

Running Tests
-------------

.. code-block:: bash

    pytest tests/

Contributing
------------

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests
5. Submit a pull request

Reporting Issues
----------------

* https://github.com/lichao312214129/HABIT/issues

Design Patterns
---------------

Dependency Injection
^^^^^^^^^^^^^^^^^^^^

HABIT uses dependency injection for better testability:

.. code-block:: python

    class ModelComparison:
        def __init__(
            self,
            config,
            evaluator: MultifileEvaluator,
            reporter: ReportExporter,
            logger,
        ):
            self.config = config
            self.evaluator = evaluator
            self.reporter = reporter
            self.logger = logger

Service Configurator
^^^^^^^^^^^^^^^^^^^^

Complex services are created via ServiceConfigurator:

.. code-block:: python

    configurator = ServiceConfigurator(config)
    model_comparison = configurator.create_model_comparison()
