Installation
============

**Windows (recommended)**: Download the portable pack → extract → double-click ``setup_habit.bat`` .

**macOS / Linux / developers**: See **Source install** below.

Demo data: :doc:`quickstart` .

Portable pack
-------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Pack
     - Notes
   * - CPU (recommended)
     - ``HABIT-win-py310-cpu-v0.1.0.tar.gz`` · `Download CPU pack <|cpu_pack_link|>`_ · code ``|cpu_pack_code|``
   * - GPU bundle
     - ~3 GB, NVIDIA GPU optional · `Download GPU pack <|gpu_pack_link|>`_ · code ``|gpu_pack_code|``

Windows portable steps
----------------------

.. note::

   Example path ``D:\habit-cpu`` — use your own short path without spaces or non-ASCII characters.

1. Extract the archive so ``python.exe`` and ``setup_habit.bat`` sit in the same folder.
2. Run ``setup_habit.bat`` , open a **new** terminal, run ``habit --version`` .
3. Continue with :doc:`quickstart` or :doc:`../gui/index` .

**Upgrade CPU pack to GPU torch** (optional): download ``torch-2.4.0+cu121-cp310-cp310-win_amd64.whl`` (`Download wheel <|torch_wheel_link|>`_ , code ``|torch_wheel_code|`` ), place next to ``python.exe`` , run ``install_gpu_torch.bat`` .

Source install
--------------

.. code-block:: bash

   conda create -n habit python=3.10
   conda activate habit
   cd /path/to/habit_project_v1
   pip install -r requirements.txt
   pip install -e .
   habit --version

Windows may also need ``pip install pyradiomics-3.0.1-cp310-cp310-win_amd64.whl`` . Troubleshooting: :doc:`../troubleshooting/faq` .

Next → :doc:`quickstart`
