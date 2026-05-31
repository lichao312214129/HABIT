安装指南
========

面向临床与研究人员：从零安装 **Miniconda 或 Anaconda**，创建 **Python 3.10** 环境 ``habit``，再安装 HABIT。无需 IDE，在 **Anaconda Prompt**（Windows）或系统终端中即可完成。

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

安装完成后，从开始菜单打开 **Anaconda Prompt** 或 **Anaconda Powershell Prompt**（任选其一）：

.. code-block:: text

   [开始] 搜索: Anaconda Prompt
   > Anaconda Prompt
   > Anaconda Powershell Prompt

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

.. code-block:: bash

   conda create -n habit python=3.10 -y
   conda activate habit

**中国大陆用户（可选，加速 pip）**：

.. code-block:: bash

   pip config set global.timeout 6000
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   pip config set global.extra-index-url https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
   pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

**获取源码**

- **ZIP（推荐普通用户）**：`下载 main.zip <https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip>`_，**直接解压**到任意位置（如 ``D:\``），无需新建或重命名文件夹；解压后会自动得到 **`HABIT-main`** 目录。
- **Git 克隆**：``git clone --depth 1 https://github.com/lichao312214129/HABIT.git``，目录名为 ``HABIT``。

进入含 ``requirements.txt`` 的目录后安装（ZIP 用户示例）：

.. code-block:: bash

   cd D:\HABIT-main
   pip install -r requirements.txt
   pip install -e .

目录结构示意（ZIP 解压后）：

.. code-block:: text

   D:\HABIT-main\
   ├── config\
   ├── habit\
   ├── requirements.txt
   └── setup.py

``requirements.txt`` 含 ``numpy==1.26.1`` 与 GPU 版 ``torch==2.4.0+cu121``（CUDA 12.1，py310 环境）。**无 NVIDIA GPU 或 macOS**：先注释文件末尾 3 行（``--extra-index-url`` 与 ``torch==...``），再执行：

.. code-block:: bash

   pip install -r requirements.txt
   pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
   pip install -e .

验证安装
--------

.. code-block:: bash

   habit --version

更新与卸载
----------

**Git 用户**：

.. code-block:: bash

   conda activate habit
   cd D:\HABIT
   git pull
   pip install -r requirements.txt --upgrade
   pip install -e .

**ZIP 用户**：重新下载并解压（仍会得到 ``HABIT-main``），``cd`` 进入该目录后重复 ``pip install -r requirements.txt`` 与 ``pip install -e .``。

卸载：``pip uninstall HABIT -y``（不删除源码与其它依赖）。

常见问题
--------

- 必须使用 **``conda create -n habit python=3.10``**；其它版本易与 PyTorch 等冲突
- 安装失败：``pip install --upgrade pip``；或按 ``requirements.txt`` 逐行 ``pip install``
- 网络超时：配置上文清华镜像；仍失败请 `提交 Issue <https://github.com/lichao312214129/HABIT/issues>`_ 或邮件 **lichao19870617@163.com**

下一步
------

:doc:`quickstart_zh` — 下载 demo 并跑通完整流程。
