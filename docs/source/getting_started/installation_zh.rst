安装指南
========

面向临床与研究人员。**Windows 普通用户** 推荐 **便携包（conda pack）** ：无需安装 Miniconda、无需 pip，解压后运行 ``setup_habit.bat`` 即可。**开发者** 或 **macOS / Linux** 用户请见下文 **方式二：源码安装**。

系统要求
---------

- **Python 3.10**（便携包已内置；源码安装须自行创建 py310 环境）
- Windows / macOS / Linux；建议内存 8 GB 及以上
- **GPU 便携包**（``HABIT-win-py310-gpu-v0.1.0``）须 **64 位 Windows** + **NVIDIA 显卡**（驱动支持 CUDA 12.1）

方式一：Windows 便携安装（推荐，普通用户）
------------------------------------------

便携包由 **conda pack** 打成，内含 Python 3.10、HABIT 及全部依赖（含 **GPU 版 PyTorch**）。**不需要** 安装 Miniconda / Anaconda，**不需要** 执行 ``pip install``。

下载（百度网盘）
~~~~~~~~~~~~~~~~

**HABIT 便携环境**（约数 GB，已内含 ``setup_habit.bat`` 、``install_gpu_torch.cmd`` 及外部工具）：

- 文件名：``HABIT-win-py310-gpu-v0.1.0.tar.gz``
- `百度网盘 <https://pan.baidu.com/s/1xaMy69z-2dZH4nFEwhd4tg?pwd=fxnh>`_ ，提取码 **fxnh**

解压与目录
~~~~~~~~~~

1. 将 ``HABIT-win-py310-gpu-v0.1.0.tar.gz`` 解压到 **任意目录**（路径宜 **短**、**无中文**、**无空格**），例如 ``D:\HABIT`` 。
2. Windows 10/11 可在资源管理器中用 **Bandizip / 7-Zip** 解压；或在命令提示符中（路径按实际修改）：

   .. code-block:: bash

      mkdir D:\HABIT
      tar -xf HABIT-win-py310-gpu-v0.1.0.tar.gz -C D:\HABIT

3. 解压后 pack 根目录应含 ``python.exe`` 、``setup_habit.bat`` 、``install_gpu_torch.cmd`` 、``Scripts\`` 等。

一键配置
~~~~~~~~

1. **双击** ``setup_habit.bat`` ，或在命令提示符中 ``cd`` 到 pack 根目录后执行：

   .. code-block:: bash

      setup_habit.bat

2. 脚本会自动：修复便携环境路径（``conda-unpack``）、将 ``Scripts\`` 与 pack 根目录 **添加到用户 PATH 最前面**（优先于本机旧版 ``habit`` ）。
3. 出现 **Press any key to exit...** 时按任意键关闭窗口。

验证
~~~~

**必须关闭当前窗口，重新打开** 新的命令提示符或 Cmder，然后执行：

.. code-block:: bash

   where habit
   habit --version

- ``where habit`` 第一行应指向 pack 内 ``...\Scripts\habit.exe`` 。
- ``habit --version`` 应显示 ``HABIT, version 0.1.0`` 。

可选：检查 GPU 版 PyTorch（便携包已含 GPU 版，一般无需再装）：

.. code-block:: bash

   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

期望类似 ``2.4.0+cu121 True`` （须 NVIDIA 驱动正常）。

配置模板与其它文件
~~~~~~~~~~~~~~~~~~

- **示例 YAML**：便携包 **不含** 仓库根目录的 ``config/`` 。请从 `GitHub 下载 ZIP <https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip>`_ 解压，将其中 ``config/`` 复制到任意工作目录，按 ``config/README_CONFIG.md`` 修改路径；字段说明见 :doc:`../configuration_zh` 。
- **随包外部工具**（官方 GPU 便携包已内置）：``Scripts\`` 中含 **dcm2niix.exe** 、**elastix.exe** 、**transformix.exe** 及 **Par0040affine.txt** 。运行 ``setup_habit.bat`` 后 ``Scripts\`` 已在 PATH **最前**，YAML 中 **可省略** ``dcm2niix_path`` 、``elastix_path`` 、``transformix_path`` （留空即走 PATH）。``elastix_parameter_files`` 仍须写 **文件路径** ，例如相对配置文件：``../../Scripts/Par0040affine.txt`` （按你的 ``config/`` 位置调整），或写 pack 内绝对路径。
- **演示数据**：见 :doc:`quickstart_zh` 中的 ``demo_data.rar`` 网盘链接。

维护者：打包前 staging（可选阅读）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在 **conda pack 之前** 运行 ``conda_pack/stage_external_tools.bat`` ，一次性复制：

- **pack 根目录**：``setup_habit.bat`` 、``install_gpu_torch.cmd`` 、``requirements-gpu-torch-only.txt``
- **Scripts\\** ：``demo_data\`` 下的 ``dcm2niix.exe`` 、``elastix.exe`` 、``transformix.exe`` 、``Par0040affine.txt``

.. code-block:: bash

   conda activate habit
   conda_pack\stage_external_tools.bat
   conda pack -n habit -o HABIT-win-py310-gpu-v0.1.0.tar.gz

便携包更新与卸载
~~~~~~~~~~~~~~~~

- **更新版本**：删除旧 pack 文件夹，重新下载新版 ``.tar.gz`` 解压，再运行 ``setup_habit.bat`` （会再次把新路径置于 PATH 最前）。
- **卸载**：删除 pack 文件夹；在 Windows「环境变量」中从用户 **Path** 里移除对应的两条路径即可（无需 ``pip uninstall``）。

.. note::

   若解压后缺少 ``setup_habit.bat`` （旧版包），可从仓库 ``conda_pack/`` 复制到 pack 根目录，或 `单独下载 <https://pan.baidu.com/s/14VlfMIjhnJy_ppNgJ5whEQ?pwd=akkw>`_ （提取码 **akkw**）。请在 **pack 根目录**（含 ``python.exe``）运行，不要对源码树 ``conda_pack\`` 直接运行。

方式二：源码安装（开发者 / macOS / Linux）
------------------------------------------

以下步骤需先安装 **Miniconda 或 Anaconda**，创建 **Python 3.10** 环境 ``habit`` ，再 ``pip`` 安装 HABIT。Windows 请在 **Anaconda Powershell Prompt** 中操作；macOS / Linux 用系统终端。

一、安装 Miniconda 或 Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若已安装其中任一发行版，可跳过本节，直接从第二节开始。

**Miniconda** 与 **Anaconda** 二选一即可；Miniconda 体积更小，推荐新用户。

Miniconda（推荐）
~~~~~~~~~~~~~~~~~

1. 打开 `Miniconda 官方下载 <https://docs.anaconda.com/miniconda>`_，选择与您系统匹配的安装包。
2. Windows 用户也可直接下载：

   `Miniconda3 Windows x86_64 <https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_

3. **Windows** ：双击 ``.exe`` ，按向导一路 **Next / 继续** 完成安装。
4. **macOS** ：下载 ``.pkg`` 安装；**Linux** ：按官方说明运行安装脚本。

Anaconda（备选）
~~~~~~~~~~~~~~~~

若更习惯 Anaconda 全家桶，从 `Anaconda 下载页 <https://www.anaconda.com/download>`_ 下载并安装，后续打开终端、创建环境与安装 HABIT 的步骤与 Miniconda **完全相同** 。

打开终端
~~~~~~~~

**Windows**

从开始菜单打开 **Anaconda Powershell Prompt** （不要用 Anaconda Prompt）：

.. code-block:: text

   [开始] 搜索: Anaconda Powershell Prompt

**macOS / Linux**

打开系统 **Terminal（终端）** 。macOS 首次安装 Miniconda/Anaconda 后请执行：

.. code-block:: bash

   conda init
   # 关闭并重新打开终端

验证 conda 是否可用
~~~~~~~~~~~~~~~~~~~

命令行前缀出现 ``(base)`` 即表示 conda 已就绪，例如：

.. code-block:: text

   (base) C:\Users\YourName>_
   (base) user@host:~$

二、创建 py310 环境并安装 HABIT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若尚未创建过 ``habit`` 环境，执行：

.. code-block:: bash

   conda create -n habit python=3.10

若出现 ``Proceed ([y]/n)?`` 或类似确认，输入 ``y`` 回车即可。

**若您之前已经创建过名为 ``habit`` 的 Python 3.10 环境** （例如重装过 HABIT、或按旧文档装过一遍），**不必再执行** ``conda create`` ，直接激活即可：

.. code-block:: bash

   conda activate habit

可用 ``conda env list`` 查看是否已有 ``habit`` 环境。

**中国大陆用户（可选，加速 pip）** ：以下四条 **只需配置一次** （写入本机 pip 全局设置）；若此前已设置过清华镜像，可跳过。

.. code-block:: bash

   pip config set global.timeout 6000
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   pip config set global.extra-index-url https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
   pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

**获取源码**

- **ZIP（推荐普通用户）** ：`下载 main.zip <https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip>`_，**直接解压** 到任意位置（Windows 如 ``D:\`` ，macOS 如「下载」），无需新建或重命名；解压后自动得到 ``HABIT-main`` 。
- **Git 克隆** ： ``git clone --depth 1 https://github.com/lichao312214129/HABIT.git`` ，目录名为 ``HABIT``。

进入项目目录并安装
~~~~~~~~~~~~~~~~~~

在 **Anaconda Powershell Prompt** 中，先找到项目根的 **完整路径** （打开该文件夹后应能直接看到 ``config`` 、``habit`` 、``requirements.txt``）。
``config/`` 目录内含各流程 **可参考的示例 YAML** （场景索引见 ``config/README_CONFIG.md`` ），与 :doc:`../configuration_zh` 配置参考配合使用。

- **Windows** ：资源管理器中进入 ``HABIT-main`` → 点击窗口上方 **地址栏** → 复制整段路径（示例 ``D:\HABIT-main``），在终端用 ``cd`` 进入该目录（路径含空格时用英文双引号包裹，例如 ``cd "D:\HABIT-main"``）。
- **macOS** ：Finder 中把 ``HABIT-main`` **拖入** 终端窗口，会自动填入完整路径；或进入该文件夹后于终端执行 ``pwd`` 查看。

**Windows** （``cd`` 到项目根，须能看到 ``pyradiomics-3.0.1-cp310-cp310-win_amd64.whl``；下文 ``D:\HABIT-main`` **只是示例路径** ，请换成你电脑上解压后的实际目录）：

.. note::

   该 ``.whl`` 为 **Windows 64 位（win_amd64）+ Python 3.10（cp310）** 预编译包，与上文 ``conda create -n habit python=3.10`` 一致。
   文件名含义： ``3.0.1`` = PyRadiomics 版本；``cp310`` = CPython 3.10；``win_amd64`` = 64 位 Windows。
   **不适用** 于 Python 3.9/3.11、32 位 Windows 或 macOS/Linux（后者请 ``pip install pyradiomics``）。

.. code-block:: bash

   conda activate habit
   cd "D:\HABIT-main"    # 仅为示例；请改为你本机 HABIT 项目根目录的完整路径（见上文复制地址栏）

   pip install -r requirements.txt
   # 预编译 wheel：仅限 Windows 64 位 + Python 3.10，无需本机 C++ 编译器
   pip install pyradiomics-3.0.1-cp310-cp310-win_amd64.whl
   pip install -e .

**macOS / Linux** ：

.. code-block:: bash

   conda activate habit
   cd ~/Downloads/HABIT-main    # 仅为示例；请改为你本机项目根目录的完整路径

   pip install -r requirements.txt
   pip install pyradiomics
   pip install -e .

``requirements.txt`` 已包含 **CPU 版** PyTorch（``torch==2.4.0+cpu`` ，阿里云镜像）。**有 NVIDIA GPU 且需加速** 时，Windows 用户可优先用下文 **网盘 wheel** 安装 GPU 版 torch；其它平台或无法使用网盘时，执行 ``pip install -r requirements-gpu.txt`` ，再 ``pip install -e .`` （见下文）。

安装完依赖后，可在终端检查 PyTorch（**不是 GPU 也能正常用 HABIT** ，只是部分步骤会慢一些）：

.. code-block:: bash

   python -c "import torch; print('torch', torch.__version__); print('CUDA available', torch.cuda.is_available())"

- ``CUDA available True``：已安装 ``requirements-gpu.txt``；``torch`` 版本通常带 ``cu121`` 后缀（例如 ``torch==2.4.0+cu121``，见下文「可选：GPU 版 PyTorch」）。
- ``CUDA available False`` ：默认 ``requirements.txt`` 中的 CPU 版 torch；HABIT 仍可用，部分计算会慢一些。

.. warning:: ZIP 解压后可能出现 ``HABIT-main`` 嵌套

   部分解压软件会多出一层 ``HABIT-main/HABIT-main/``。若当前目录里 **只有** 子文件夹 ``HABIT-main`` 、没有 ``config`` 和 ``habit`` ，请再进入内层，重新复制地址栏的 **完整路径** 后再 ``cd`` （例如 ``cd "D:\HABIT-main\HABIT-main"``）。

目录结构示意（ZIP 解压后，文件夹名均为 ``HABIT-main``）：

.. code-block:: text

   HABIT-main/
   ├── config/
   ├── habit/
   ├── pyradiomics-3.0.1-cp310-cp310-win_amd64.whl
   ├── torch-2.4.0+cu121-cp310-cp310-win_amd64.whl   # 可选：从网盘下载的 GPU 版 torch
   ├── requirements.txt
   ├── requirements-gpu.txt
   └── setup.py

可选：GPU 版 PyTorch（仅 NVIDIA 显卡）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

默认 ``pip install -r requirements.txt`` 会安装 **CPU** 版 ``torch==2.4.0+cpu`` （阿里云镜像，体积较小）。**有 NVIDIA GPU** 且需要 TorchRadiomics / 生境 GPU 加速时，须在 ``pip install -e .`` **之前** 将 torch 换为 **CUDA 12.1** 的 ``torch==2.4.0+cu121`` （约 2 GB）。**无独显或 macOS** 用户无需执行本节。

**方式一：网盘 wheel（Windows 推荐，避免 pip 在线下载过慢）**

GPU 版 torch 体积大，``pip install -r requirements-gpu.txt`` 从镜像在线拉取可能 **很慢或中断** 。维护者在百度网盘提供了 **Windows 64 位（win_amd64）+ Python 3.10（cp310）** 预下载 wheel（与 ``conda create -n habit python=3.10`` 一致；**不适用** 于 Python 3.9/3.11 或 macOS/Linux）：

- `百度网盘 <https://pan.baidu.com/s/1eY4lmNegCYh5KgQB640FmA?pwd=nt7k>`_ ，提取码 **nt7k**
- 下载 ``torch-2.4.0+cu121-cp310-cp310-win_amd64.whl``（约 2 GB）

**wheel 放哪里（必读，否则 ``pip install`` 会报找不到文件）**

网盘文件默认在 **「下载」** 文件夹；``pip install torch-2.4.0+cu121-...whl`` 会在 **当前终端所在目录** 查找该文件，不会自动去「下载」里找。

请任选一种方式：

1. **推荐：复制到 HABIT 项目根目录**（与 ``requirements.txt`` 、``config`` 、``habit`` 文件夹 **同级**），再在终端 ``cd`` 到该目录后安装。目录示意见上文「目录结构示意」。
2. **不移动文件：在 ``pip install`` 里写 wheel 的完整路径** ，例如网盘下到「下载」时：

   .. code-block:: bash

      pip install "C:\Users\YourName\Downloads\torch-2.4.0+cu121-cp310-cp310-win_amd64.whl"

   路径含空格时用英文双引号包裹；``YourName`` 与 ``Downloads`` 请改为你本机实际路径。

**如何确认目录正确** ：在终端 ``cd`` 到项目根后，Windows 可执行 ``dir torch-2.4.0+cu121-cp310-cp310-win_amd64.whl`` ，应能看到该文件；若提示找不到，说明 ``cd`` 位置不对，或 ``.whl`` 仍在「下载」等其它文件夹。

在 **Python 3.10** 的 ``habit`` 环境中执行（**先** ``requirements.txt`` ，**再** 装 wheel，**最后** ``pip install -e .``；下例假定 ``.whl`` 已放在项目根，与 ``pyradiomics`` wheel 相同做法）：

.. code-block:: bash

   conda activate habit
   cd "D:\HABIT-main"    # 仅为示例；请改为你本机项目根目录的完整路径（须能看到 requirements.txt 与 .whl）

   dir torch-2.4.0+cu121-cp310-cp310-win_amd64.whl    # 确认文件在当前目录；找不到则先 cd 或改用完整路径

   pip install -r requirements.txt
   pip install torch-2.4.0+cu121-cp310-cp310-win_amd64.whl
   python -c "import torch; print('torch', torch.__version__); print('CUDA available', torch.cuda.is_available())"
   pip install -e .

若提示找不到 ``.whl`` ：确认 ``cd`` 在含 ``config`` 、``habit`` 的项目根；若 ZIP 多嵌套一层 ``HABIT-main`` ，进入内层再安装；或改用上文 **完整路径** 形式的 ``pip install "D:\...\torch-2.4.0+cu121-cp310-cp310-win_amd64.whl"`` 。

**方式二：pip 在线安装（Windows / Linux，或无法使用网盘时）**

.. code-block:: bash

   pip install -r requirements-gpu.txt

若环境中 **已是** ``torch==2.4.0+cu121``，pip 会显示已满足、不会重复下载。

若刚执行完 ``requirements.txt`` 后 ``torch`` 仍为 ``+cpu`` 且 ``CUDA available False``，可先加 ``--upgrade`` 再装一次；**仅当** pip 仍未替换为 GPU 版时，再尝试 ``--force-reinstall`` （会强制重装，已有正确版本时不必使用）。

三、验证安装（源码方式）
------------------------

.. code-block:: bash

   habit --version

四、更新与卸载（源码方式）
--------------------------

**Git 用户** ：

.. code-block:: bash

   conda activate habit
   cd D:\HABIT              # Windows
   # cd ~/Projects/HABIT    # macOS / Linux 示例
   git pull
   pip install -r requirements.txt --upgrade
   pip install -e .

**ZIP 用户** ：重新下载并解压（仍会得到 ``HABIT-main``），``cd`` 到该目录后重复 ``pip install -r requirements.txt`` （GPU 用户再加 ``requirements-gpu.txt``）与 ``pip install -e .``。

卸载： ``pip uninstall HABIT -y`` （不删除源码与其它依赖）。

五、常见问题
------------

**便携包（方式一）**

- ``setup_habit.bat`` 报错找不到 ``python.exe`` ：确认 bat 与 ``python.exe`` 在同一文件夹；不要运行源码树里 ``conda_pack\`` 下的副本。
- 运行 ``setup_habit.bat`` 后 ``habit`` 仍指向旧路径：须 **新开终端** ；或再次运行 ``setup_habit.bat``（会把 pack 路径移到 PATH **最前**）。
- ``CUDA available False`` ：检查 NVIDIA 驱动；便携 GPU 包已含 ``torch+cu121``，一般 **无需** 再执行 GPU 安装脚本。
- ``conda-unpack`` 相关提示：若 ``habit --version`` 正常可忽略；若无法运行 habit，请重新解压到新文件夹再执行 setup。

**源码安装（方式二）**

- 环境须为 **Python 3.10** ，conda 环境名建议 ``habit``；首次安装执行 ``conda create`` ，已有 ``habit`` 环境则只 ``conda activate habit``
- 安装失败： ``pip install --upgrade pip``；或按 ``requirements.txt`` 逐行 ``pip install``
- 网络超时：配置上文清华镜像；仍失败请 `提交 Issue <https://github.com/lichao312214129/HABIT/issues>`_ 或邮件 **lichao19870617@163.com**

六、Windows：安装 ``pyradiomics``（源码方式）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**推荐（ZIP / 源码包自带 wheel）**

HABIT 主目录（与 ``requirements.txt`` 同级）自带 **预编译** wheel（已在 **Windows 64 位 + Python 3.10** 环境下编好，无需本机 C++ 编译器）：

``pyradiomics-3.0.1-cp310-cp310-win_amd64.whl``

- ``cp310`` ：仅适用于 **Python 3.10** （与 ``habit`` 环境一致）
- ``win_amd64`` ：仅适用于 **64 位 Windows** （常见台式机/笔记本）
- 若为 **Python 3.9/3.11**、**32 位 Windows** 或其它系统，不能使用此文件，请参考下文备选方案

在 **Python 3.10** 的 ``habit`` 环境中，于项目根目录执行（**先** ``requirements.txt`` 安装 ``numpy`` 等依赖，**再** 装 wheel）：

.. code-block:: bash

   conda activate habit
   cd "D:\HABIT-main"    # 仅为示例；请改为你本机项目根目录的完整路径
   pip install -r requirements.txt
   pip install pyradiomics-3.0.1-cp310-cp310-win_amd64.whl
   python -c "import radiomics; print('pyradiomics OK')"

``requirements.txt`` 已含 ``numpy==1.26.1`` ，**不包含** ``pyradiomics`` （避免 pip 从源码编译）。顺序：先 ``pip install -r requirements.txt`` ，再装 wheel，最后 ``pip install -e .``。

若提示找不到 ``.whl`` 文件：确认 ``cd`` 在含 ``config`` 、``habit`` 的项目根；若 ZIP 多嵌套一层 ``HABIT-main`` ，进入内层再安装。

**备选：wheel 安装失败时从源码安装（需 C++ 编译器）**

若 ``pip install ...whl`` 报错，或包内无该 wheel（如仅 Git 克隆），可在安装 **Microsoft C++ Build Tools** 后从源码安装。**不要** 使用普通 ``pip install pyradiomics`` （易因 pip 构建隔离导致缺 ``numpy`` 或仍缺编译器而失败），请使用 ``--no-build-isolation`` ，并确保已先执行 ``pip install -r requirements.txt`` （环境中已有 ``numpy``）。

1. 安装 `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_，工作负载勾选 **使用 C++ 的桌面开发** （**Desktop development with C++**）；安装后 **关闭并重新打开** 终端。
2. 在项目根目录执行：

   .. code-block:: bash

      conda activate habit
      cd "D:\HABIT-main"    # 仅为示例；请改为你本机项目根目录的完整路径
      pip install -r requirements.txt
      pip install pyradiomics --no-build-isolation
      python -c "import radiomics; print('pyradiomics OK')"
      pip install -e .

说明： ``--no-build-isolation`` 让构建过程使用当前环境里已安装的 ``numpy`` 等包；仍须本机 **MSVC** 编译 ``pyradiomics`` 的 C 扩展。亦可尝试 ``conda install -c conda-forge pyradiomics``。

**误用** ``pip install pyradiomics`` （无 ``--no-build-isolation``）时，常见 ``No module named 'numpy'`` 或 ``Microsoft Visual C++ 14.0 or greater is required``；请改用上文 wheel 或备选命令。

**macOS / Linux**

无自带 Windows wheel；先 ``pip install -r requirements.txt`` ，再 ``pip install pyradiomics`` （需 **Python 3.10** ）。部分环境仍需 C++ 工具链或 conda 预编译包。

下一步
------

:doc:`quickstart_zh` — 下载 demo 并跑通完整流程。
