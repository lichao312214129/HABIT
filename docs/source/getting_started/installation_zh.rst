安装指南
========

面向临床与研究人员：从零安装 **Miniconda 或 Anaconda**，创建 **Python 3.10** 环境 ``habit``，再安装 HABIT。Windows 请在 **Anaconda Powershell Prompt** 中操作；macOS / Linux 用系统终端。

系统要求
---------

- **Python 3.10**（与项目 ``requirements.txt`` / py310 测试环境一致）
- Windows / macOS / Linux；建议内存 8 GB 及以上

一、安装 Miniconda 或 Anaconda
-------------------------------

若已安装其中任一发行版，可跳过本节，直接从第二节开始。

**Miniconda** 与 **Anaconda** 二选一即可；Miniconda 体积更小，推荐新用户。

Miniconda（推荐）
~~~~~~~~~~~~~~~~~

1. 打开 `Miniconda 官方下载 <https://docs.anaconda.com/miniconda>`_，选择与您系统匹配的安装包。
2. Windows 用户也可直接下载：

   `Miniconda3 Windows x86_64 <https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_

3. **Windows**：双击 ``.exe``，按向导一路 **Next / 继续** 完成安装。
4. **macOS**：下载 ``.pkg`` 安装；**Linux**：按官方说明运行安装脚本。

Anaconda（备选）
~~~~~~~~~~~~~~~~

若更习惯 Anaconda 全家桶，从 `Anaconda 下载页 <https://www.anaconda.com/download>`_ 下载并安装，后续打开终端、创建环境与安装 HABIT 的步骤与 Miniconda **完全相同**。

打开终端
~~~~~~~~

**Windows**

从开始菜单打开 **Anaconda Powershell Prompt**（不要用 Anaconda Prompt）：

.. code-block:: text

   [开始] 搜索: Anaconda Powershell Prompt

**macOS / Linux**

打开系统 **Terminal（终端）**。macOS 首次安装 Miniconda/Anaconda 后请执行：

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
-------------------------------

若尚未创建过 ``habit`` 环境，执行：

.. code-block:: bash

   conda create -n habit python=3.10

若出现 ``Proceed ([y]/n)?`` 或类似确认，输入 ``y`` 回车即可。

**若您之前已经创建过名为 ``habit`` 的 Python 3.10 环境**（例如重装过 HABIT、或按旧文档装过一遍），**不必再执行** ``conda create``，直接激活即可：

.. code-block:: bash

   conda activate habit

可用 ``conda env list`` 查看是否已有 ``habit`` 环境。

**中国大陆用户（可选，加速 pip）**：以下四条**只需配置一次**（写入本机 pip 全局设置）；若此前已设置过清华镜像，可跳过。

.. code-block:: bash

   pip config set global.timeout 6000
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   pip config set global.extra-index-url https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
   pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

**获取源码**

- **ZIP（推荐普通用户）**：`下载 main.zip <https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip>`_，**直接解压**到任意位置（Windows 如 ``D:\``，macOS 如「下载」），无需新建或重命名；解压后自动得到 **`HABIT-main`**。
- **Git 克隆**：``git clone --depth 1 https://github.com/lichao312214129/HABIT.git``，目录名为 ``HABIT``。

进入项目目录并安装
~~~~~~~~~~~~~~~~~~

在 **Anaconda Powershell Prompt** 中，先找到项目根的**完整路径**（打开该文件夹后应能直接看到 ``config``、``habit``、``requirements.txt``）：

- **Windows**：资源管理器中进入 ``HABIT-main`` → 点击窗口上方**地址栏** → 复制整段路径（示例 ``D:\HABIT-main``），在终端执行 ``cd "粘贴的路径"``（路径含空格时必须加引号）。
- **macOS**：Finder 中把 ``HABIT-main`` **拖入**终端窗口，会自动填入完整路径；或进入该文件夹后于终端执行 ``pwd`` 查看。

**Windows**（``cd`` 到项目根，须能看到 ``pyradiomics-3.0.1-cp310-cp310-win_amd64.whl``）：

.. code-block:: bash

   conda activate habit
   cd "D:\HABIT-main"

   pip install numpy==1.26.1
   pip install pyradiomics-3.0.1-cp310-cp310-win_amd64.whl
   pip install -r requirements.txt
   pip install -r requirements-cpu.txt
   pip install -e .

**macOS / Linux**：

.. code-block:: bash

   conda activate habit
   cd ~/Downloads/HABIT-main

   pip install numpy==1.26.1
   pip install pyradiomics
   pip install -r requirements.txt
   pip install -r requirements-cpu.txt
   pip install -e .

有 NVIDIA GPU 时，将 ``requirements-cpu.txt`` 换成 ``requirements-gpu.txt``（见下文「安装 PyTorch」）。

安装完依赖后，可在终端检查 PyTorch（**未装 torch 时部分功能不可用**；**不是 GPU 也能正常用 HABIT**，只是部分步骤会慢一些）：

.. code-block:: bash

   python -c "import torch; print('torch', torch.__version__); print('CUDA available', torch.cuda.is_available())"

- ``CUDA available True``：已安装 ``requirements-gpu.txt``（版本号常含 ``+cu121``）。
- ``CUDA available False``：已安装 ``requirements-cpu.txt`` 或未装 GPU 版；HABIT 仍可用，部分计算会慢一些。

.. warning:: ZIP 解压后可能出现 ``HABIT-main`` 嵌套

   部分解压软件会多出一层 ``HABIT-main/HABIT-main/``。若当前目录里**只有**子文件夹 ``HABIT-main``、没有 ``config`` 和 ``habit``，请再进入内层，重新复制地址栏的**完整路径**后再 ``cd``（例如 ``cd "D:\HABIT-main\HABIT-main"``）。

目录结构示意（ZIP 解压后，文件夹名均为 ``HABIT-main``）：

.. code-block:: text

   HABIT-main/
   ├── config/
   ├── habit/
   ├── pyradiomics-3.0.1-cp310-cp310-win_amd64.whl
   ├── requirements.txt
   ├── requirements-cpu.txt
   ├── requirements-gpu.txt
   └── setup.py

安装 PyTorch（``requirements.txt`` 不含 torch）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``requirements.txt`` 只安装 HABIT 的常规依赖（体积大、耗时的 **PyTorch 单独安装**）。在 ``pip install -r requirements.txt`` 之后、``pip install -e .`` 之前，**必须二选一** 安装 PyTorch（均使用 **阿里云** wheels 镜像）：

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 用户类型
     - 命令
     - 说明
   * - **大多数用户（推荐）**
     - ``pip install -r requirements-cpu.txt``
     - CPU 版 ``torch==2.4.0+cpu``，体积较小，无 NVIDIA 显卡或不需要 GPU 加速时选此项
   * - **有 NVIDIA GPU 且需加速**
     - ``pip install -r requirements-gpu.txt``
     - CUDA 12.1 的 ``torch==2.4.0+cu121``，约 2 GB；**不要** 再装 ``requirements-cpu.txt``
   * - **先装 CPU 再换 GPU**
     - 先 ``requirements-cpu.txt``，再 ``pip install -r requirements-gpu.txt --upgrade --force-reinstall``
     - 仅当已误装 CPU 版、需要改用 GPU 时

**macOS / 无独显 Windows**：安装 ``requirements-cpu.txt`` 即可，不要安装 ``requirements-gpu.txt``。

若跳过上述两步直接 ``pip install -e .``，环境中可能没有 ``torch``，生境 GPU / TorchRadiomics 相关功能将无法使用。

验证安装
--------

.. code-block:: bash

   habit --version

更新与卸载
----------

**Git 用户**：

.. code-block:: bash

   conda activate habit
   cd D:\HABIT              # Windows
   # cd ~/Projects/HABIT    # macOS / Linux 示例
   git pull
   pip install -r requirements.txt --upgrade
   pip install -r requirements-cpu.txt --upgrade
   pip install -e .

**ZIP 用户**：重新下载并解压（仍会得到 ``HABIT-main``），``cd`` 到该目录后重复 ``pip install -r requirements.txt``、``requirements-cpu.txt``（或 ``requirements-gpu.txt``）与 ``pip install -e .``。

卸载：``pip uninstall HABIT -y``（不删除源码与其它依赖）。

常见问题
--------

- 环境须为 **Python 3.10**，conda 环境名建议 ``habit``；首次安装执行 ``conda create``，已有 ``habit`` 环境则只 ``conda activate habit``
- 安装失败：``pip install --upgrade pip``；或按 ``requirements.txt`` 逐行 ``pip install``
- 网络超时：配置上文清华镜像；仍失败请 `提交 Issue <https://github.com/lichao312214129/HABIT/issues>`_ 或邮件 **lichao19870617@163.com**

Windows：安装 ``pyradiomics``（免 C++ 编译器）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**推荐（ZIP / 源码包自带 wheel）**

HABIT 主目录（与 ``requirements.txt`` 同级）自带预编译文件：

``pyradiomics-3.0.1-cp310-cp310-win_amd64.whl``

在 **Python 3.10** 的 ``habit`` 环境中，于项目根目录执行（**先于** ``pip install -r requirements.txt``）：

.. code-block:: bash

   conda activate habit
   cd "D:\HABIT-main"    # 换成你的路径；须能看到上述 .whl 文件
   pip install numpy==1.26.1
   pip install pyradiomics-3.0.1-cp310-cp310-win_amd64.whl
   python -c "import radiomics; print('pyradiomics OK')"

``requirements.txt`` **不包含** ``pyradiomics``，避免 pip 从源码编译。装完 wheel 后再执行 ``pip install -r requirements.txt`` 与其余步骤。

若提示找不到 ``.whl`` 文件：确认 ``cd`` 在含 ``config``、``habit`` 的项目根；若 ZIP 多嵌套一层 ``HABIT-main``，进入内层再安装。

**仍报错「缺少 C++ / Failed building wheel」时**

说明 pip 仍在尝试**源码编译** ``pyradiomics``（例如未先装 wheel、或误执行了 ``pip install pyradiomics``）。请改用上文 wheel 命令；勿在 ``requirements.txt`` 中自行加回 ``pyradiomics``。

若必须自行编译（无自带 wheel 的 Git 浅克隆等），可安装 `Microsoft C++ Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_，工作负载勾选 **使用 C++ 的桌面开发**（**Desktop development with C++**），安装后重启终端再 ``pip install pyradiomics``。或尝试 ``conda install -c conda-forge pyradiomics``。

**macOS / Linux**

无自带 Windows wheel；在 ``pip install -r requirements.txt`` 前或后执行 ``pip install pyradiomics``（需 **Python 3.10**）。部分环境仍需 C++ 工具链或 conda 预编译包。

下一步
------

:doc:`quickstart_zh` — 下载 demo 并跑通完整流程。
