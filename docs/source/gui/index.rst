Web GUI
=======

HABIT includes a **Gradio Web GUI** for users who prefer not to use the terminal. Tabs follow the pipeline from left to right.

Launch
------

From the project root (folder containing ``config/`` ):

.. code-block:: bash

   habit gui

Default URL: `http://127.0.0.1:8501` (browser opens automatically).

Optional:

.. code-block:: bash

   habit gui --host 127.0.0.1 --port 8501

Tabs
----

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Tab
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

Common actions
--------------

- **Pick paths** using folder / file buttons.
- **Load / save YAML** from ``config/`` templates.
- **Run / Stop** — logs stream at the bottom.
- **Open output folder** when the job finishes.

The GUI uses the **same YAML configs and backend** as the CLI.

Configuration: :doc:`../configuration/index` . Step-by-step: :doc:`../how_to/index` .

FAQ
---

**Browser does not open** — try ``habit gui --port 8502`` .

**Job failed** — read the log panel; see :doc:`../troubleshooting/faq` .
