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
   * - Updated GPU bundle
     - ``HABIT-win-py310-gpu-v0.1.0.tar-v1.gz`` and one additional file · |download_gpu_pack_v1| · code |gpu_pack_v1_code|

The updated share contains ``HABIT-win-py310-gpu-v0.1.0.tar-v1.gz`` and one additional file (extract code |gpu_pack_v1_code|).

Windows portable steps
----------------------

.. note::

   Example path ``D:\habit-cpu`` — use your own short path without spaces or non-ASCII characters.

1. Extract the archive so ``python.exe`` and ``setup_habit.bat`` sit in the same folder.
2. Run ``setup_habit.bat`` , open a **new** terminal, run ``habit --version`` .
3. Continue with :doc:`quickstart` .

**Upgrade CPU pack to GPU torch** (optional): download ``torch-2.4.0+cu121-cp310-cp310-win_amd64.whl`` (|download_torch_wheel| , code |torch_wheel_code| ), place next to ``python.exe`` , run ``install_gpu_torch.bat`` .

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
