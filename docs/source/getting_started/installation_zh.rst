安装指南
========

**Windows 用户（推荐）**：下载便携包 → 解压 → 双击 ``setup_habit.bat`` 即可。

**macOS / Linux 或需要改代码的用户**：见下文 **方式二：源码安装**。

跑 Demo 时需另下载配置与演示数据，见 :doc:`quickstart_zh`。

选哪个便携包？
--------------

**GPU 整包**（约 3 GB，有 NVIDIA 显卡且希望更快时可选）

- 文件：``HABIT-win-py310-gpu-v0.1.0.tar.gz``
- `百度网盘 <https://pan.baidu.com/s/1bzh3DvNmiL4m-Wdw7K0Tcg?pwd=8wzx>`_ ，提取码 **8wzx**

**CPU 版**（推荐，体积小）

- 文件：``HABIT-win-py310-cpu-v0.1.0.tar.gz``
- `百度网盘 <https://pan.baidu.com/s/1dG4ibQONxvMOFZm1mOKpFw?pwd=ycva>`_ ，提取码 **ycva**

- **不确定选哪个**：选 **CPU 版**（普通电脑都能用，只是部分步骤稍慢）。
- **有 NVIDIA 显卡且希望更快**：直接下载 **GPU 整包**，按下方 **两步** 安装即可，**无需** 再装 GPU 组件。

方式一：Windows 便携安装
------------------------

**说明**：无论 CPU 版还是 GPU 整包，均为 **解压 → ``setup_habit.bat``** 两步。**GPU 整包用户装完后请跳过下文「CPU 版后期升级 GPU」**。

第一步：解压
~~~~~~~~~~

1. 在 ``D:\`` 新建文件夹，例如 ``D:\habit-cpu`` （CPU 版）或 ``D:\habit-gpu`` （GPU 整包）。路径宜 **短、无中文、无空格**。
2. 把网盘下载的压缩包 **移入** 该文件夹。
3. 进入文件夹，对压缩包 **右键 → 解压到当前文件夹**（Bandizip / 7-Zip / WinRAR 均可）。
4. 确认该文件夹里 **直接有** ``python.exe`` 和 ``setup_habit.bat`` （在同一层，不要多一层子文件夹）。

   .. code-block:: text

      D:\habit-cpu\
      ├── python.exe
      ├── setup_habit.bat
      └── Scripts\

第二步：运行 setup_habit.bat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **双击** ``setup_habit.bat`` 。
2. 窗口提示 **Press any key to exit...** 时按任意键关闭。
3. **关掉窗口，重新打开** 一个新的「命令提示符」，输入：

   .. code-block:: bash

      habit --version

   应显示 ``HABIT, version 0.1.0`` 。

安装完成。跑 Demo 见 :doc:`quickstart_zh` 。

CPU 版后期升级 GPU（可跳过）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**若您下载的是 GPU 整包，本节不用看。**

仅适用于：**已装好 CPU 版**、电脑有 **NVIDIA 显卡**、又不想重新下载 3 GB GPU 整包时，可按下列步骤把 CPU 版升级为 GPU 版：

1. 从网盘下载 ``torch-2.4.0+cu121-cp310-cp310-win_amd64.whl``（约 2 GB）：
   `百度网盘 <https://pan.baidu.com/s/1eY4lmNegCYh5KgQB640FmA?pwd=nt7k>`_ ，提取码 **nt7k**
2. 把该文件 **复制到** 便携包文件夹（与 ``python.exe`` 同级，如 ``D:\habit-cpu\``）。
3. **双击** ``install_gpu_torch.bat`` 。

便携包常见问题
~~~~~~~~~~~~~~

- **habit 命令找不到**：是否 **重新打开** 了命令提示符？是否在同一文件夹运行了 ``setup_habit.bat`` ？
- **解压后没有 python.exe**：解压多了一层，把子文件夹里的内容 **全部上移** 到目标文件夹。
- **GPU 整包仍显示 CUDA False**：检查 NVIDIA 驱动是否正常。
- **CPU 版显示 CUDA False**：正常；若要加速，见上文「CPU 版后期升级 GPU」，或直接改用 **GPU 整包** 重新安装。

方式二：源码安装（macOS / Linux / 开发者）
------------------------------------------

需要自行安装 Python 环境，步骤比便携包多。Windows 用户若无特殊需求，请优先用 **方式一**。

1. 安装 Miniconda
~~~~~~~~~~~~~~~~~

下载并安装 `Miniconda <https://docs.anaconda.com/miniconda>`_ （Windows 可直接下 `安装包 <https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_ ）。

2. 打开终端
~~~~~~~~~~~

- **Windows**：开始菜单搜索 **Anaconda Powershell Prompt** 并打开。
- **macOS / Linux**：打开系统终端；首次安装 Miniconda 后执行 ``conda init`` 并重启终端。

3. 创建环境并下载源码
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n habit python=3.10
   conda activate habit

下载 `GitHub 源码 ZIP <https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip>`_ ，解压到任意位置（如 ``D:\HABIT-main`` ）。用资源管理器打开解压后的文件夹，应能看到 ``config`` 、``habit`` 、``requirements.txt`` 。

若文件夹里 **只有** 嵌套的 ``HABIT-main`` 子文件夹，请再进入 **内层** 那一级。

4. 安装 HABIT
~~~~~~~~~~~~~

在终端中 ``cd`` 到项目根目录（把资源管理器地址栏的路径复制过来即可），然后执行：

**Windows**：

.. code-block:: bash

   conda activate habit
   cd "D:\HABIT-main"

   pip install -r requirements.txt
   pip install pyradiomics-3.0.1-cp310-cp310-win_amd64.whl
   pip install -e .

**macOS / Linux**：

.. code-block:: bash

   conda activate habit
   cd ~/Downloads/HABIT-main

   pip install -r requirements.txt
   pip install pyradiomics
   pip install -e .

5. 验证
~~~~~~~

.. code-block:: bash

   habit --version

可选：GPU 版 PyTorch（源码安装 + NVIDIA 显卡）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

默认安装的是 CPU 版 PyTorch。有 NVIDIA 显卡且需要加速时，在 ``pip install -e .`` **之前**：

1. 从网盘下载 ``torch-2.4.0+cu121-cp310-cp310-win_amd64.whl``（链接见上文「CPU 版后期升级 GPU」）。
2. 将 ``.whl`` 放到项目根目录（与 ``requirements.txt`` 同级）。
3. 执行：

   .. code-block:: bash

      pip install -r requirements.txt
      pip install torch-2.4.0+cu121-cp310-cp310-win_amd64.whl
      pip install -e .

源码安装常见问题
~~~~~~~~~~~~~~~~

- **pip 很慢或超时**：可配置清华镜像（只需一次）：

  .. code-block:: bash

     pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

- **已有 habit 环境**：不必重复 ``conda create`` ，直接 ``conda activate habit`` 即可。
- **仍无法安装**： `GitHub Issues <https://github.com/lichao312214129/HABIT/issues>`_ 或邮件 **lichao19870617@163.com**

下一步
------

:doc:`quickstart_zh` — 下载配置与演示数据，跑通完整流程。
