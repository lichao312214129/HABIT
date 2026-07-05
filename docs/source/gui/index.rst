Web GUI (under development)
============================

.. note::

   The HABIT Web GUI is **under active development** and is **not yet recommended**
   for routine use. For production workflows, use the **CLI** and YAML configs
   documented in :doc:`../how_to/index` and :doc:`../configuration/index`.

Planned scope
-------------

The next-generation GUI (``habit-gui/`` in the repository) will cover the same
pipeline as the CLI:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Step
     - CLI equivalent
   * - 1. DICOM sort
     - ``habit sort-dicom``
   * - 2. Preprocessing
     - ``habit preprocess``
   * - 3. Habitat segmentation
     - ``habit get-habitat``
   * - 4. Feature extraction
     - ``habit extract``
   * - 5. Machine learning
     - ``habit model``
   * - 6. Model comparison
     - ``habit compare``

Early preview (developers)
--------------------------

Developers who want to try the work-in-progress GUI from a source checkout:

.. code-block:: bash

   cd habit-gui/web && npm install && npm run build
   habit gui

Default URL when available: ``http://127.0.0.1:8501``.

Until the GUI is officially released, prefer :doc:`../tutorial/quickstart` and
:doc:`../how_to/index` for step-by-step instructions.
