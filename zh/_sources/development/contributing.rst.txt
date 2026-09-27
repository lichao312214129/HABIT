Contributing
============

Thank you for your interest in HABIT! We welcome contributions of all kinds.

How to contribute
-----------------

Report bugs
~~~~~~~~~~~

If you find a bug, please open a GitHub issue:

1. Search existing issues to avoid duplicates.
2. Include a clear title, steps to reproduce, expected vs. actual behavior, and environment (Python version, OS).

Submit code
~~~~~~~~~~~

1. Fork the repository.
2. Create a feature branch (``git checkout -b feature/AmazingFeature``).
3. Commit your changes (``git commit -m 'Add some AmazingFeature'``).
4. Push to the branch (``git push origin feature/AmazingFeature``).
5. Open a Pull Request.

Code style
----------

* Follow PEP 8.
* Use **snake_case** for Python variables, function parameters, module-level names,
  and YAML / Pydantic configuration field keys (e.g. ``out_dir``, ``kernel_radius``,
  ``feature_construction``). Use **PascalCase** for classes only.
* Add docstrings where appropriate.
* Add tests for new behavior.
* Ensure ``pytest tests/`` passes.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Convention
     - Description
   * - Type annotations
     - Annotate function inputs and outputs explicitly, for example
       ``x_train: np.ndarray``.
   * - Comment language
     - Write detailed code comments in **English**.
   * - Plot text
     - All text generated inside programmatic plots must be **English**.
   * - Utility organization
     - Put shared utilities in ``habit/utils/`` and use
       ``habit/utils/progress_utils.py`` for progress bars.

Documentation
-------------

Documentation contributions are welcome:

* Fix typos and clarify wording.
* Add tutorials or examples.
* Keep the single English docs site up to date.

Development setup
-----------------

Use the same **Python 3.10** conda environment ``habit`` as end users (see
:doc:`../tutorial/installation`)::

   conda activate habit
   pip install -e .
   pytest tests/

How to run tests
----------------

Tests live in the repository-level ``tests/`` directory and are configured
by ``tests/pytest.ini``. There are two main categories:

1. **pytest unit and CLI tests** (``test_*.py``), using
   ``click.testing.CliRunner`` or direct API calls.
2. **Executable demo scripts**, such as
   ``tests/habitat/habitat_two_step_voxel_radiomics_train.py``. These use demo
   YAML files under ``config/`` and provide end-to-end smoke coverage.

Select tests by marker (markers are defined in ``pytest.ini``)::

   pytest tests/ -m unit
   pytest tests/ -m "habitat and not slow"
   pytest tests/ -m cli
   pytest tests/test_architecture_contracts.py -m unit

Available markers: ``slow`` / ``integration`` / ``unit`` / ``preprocessing`` /
``habitat`` / ``ml`` / ``utils`` / ``cli``.

``tests/test_architecture_contracts.py`` protects registry and orchestrator
contracts. It **must pass** after adding a factory or orchestrator. For the
rules themselves, see :doc:`architecture` (Invariants).

``tests/conftest.py`` provides fixtures such as ``project_root`` and
``demo_data_dir``; example data is stored in ``demo_data/``. End-to-end
examples are in ``tests/integration/``.

Third-party habitat plugins (registries and entry points) are documented in
:doc:`../customization/index`. Adding a CLI command or a v0.1 YAML step
schema is a maintainer task; see ``developer/sphinx_archive/dev_workflow.rst``
in the repository.

Pull requests
-------------

Before submitting a PR:

* All tests pass.
* Docs updated if behavior or config changed.
* PR description explains the change clearly.

Code of conduct
---------------

* Be respectful to all contributors.
* Accept constructive feedback.
* Focus on what helps the community.

Thank you for contributing!
