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
* Add docstrings where appropriate.
* Add tests for new behavior.
* Ensure ``pytest tests/`` passes.

Documentation
-------------

Documentation contributions are welcome:

* Fix typos and clarify wording.
* Add tutorials or examples.
* Keep the single English docs site up to date.

Development setup
-----------------

Use the same **Python 3.10** conda environment ``habit`` as end users (see :doc:`../tutorial/installation`).

.. code-block:: bash

   conda activate habit
   pip install -e ".[dev]"
   pytest tests/

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
