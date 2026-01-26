Development Guide
=================

This section contains documentation for developers working on HABIT.

.. toctree::
   :maxdepth: 2
   :caption: Development Topics

   architecture
   contributing
   design_patterns
   testing
   metrics_optimization

Overview
--------

HABIT is a comprehensive toolkit for habitat analysis in biomedical imaging. This guide covers:

* **Architecture**: System design and module organization
* **Contributing**: How to contribute to HABIT
* **Design Patterns**: Common patterns used in HABIT
* **Testing**: Testing strategies and best practices
* **Metrics Optimization**: Performance improvements and feature enhancements

Getting Started with Development
---------------------------------

1. Fork the repository
2. Create a development environment
3. Install dependencies: ``pip install -e .[dev]``
4. Run tests: ``pytest tests/``
5. Make your changes
6. Submit a pull request

Development Principles
----------------------

1. **Code Quality**: Follow PEP 8 and use type hints
2. **Testing**: Write tests for new features
3. **Documentation**: Update docs for API changes
4. **Performance**: Profile before optimizing
5. **Compatibility**: Maintain backward compatibility

Key Modules
-----------

* ``habit.core.preprocessing``: Image preprocessing pipeline
* ``habit.core.habitat_analysis``: Habitat clustering and analysis
* ``habit.core.machine_learning``: ML workflows and evaluation
* ``habit.utils``: Shared utilities (logging, progress, visualization)

Recent Improvements
-------------------

**Metrics Module Optimization (v2.0)**

* 8x performance improvement via confusion matrix caching
* Extended target metrics support (PPV, NPV, F1-score)
* Intelligent threshold selection (Pareto+Youden)
* Fallback mechanism for unattainable targets
* Category-based metric filtering

See :doc:`metrics_optimization` for details.

Contact
-------

For questions or suggestions, please open an issue on GitHub.
