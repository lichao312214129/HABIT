Optional dependencies
=====================

Install extras only when you need them, for example::

   pip install "habitat-analysis[view]"

See :doc:`/tutorial/installation` for the full list (napari / Qt view,
GPU texture, registration, AutoML, and so on). Missing optionals raise
``OptionalDependencyError`` with an install hint.
