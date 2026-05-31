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

.. code-block:: bash

   conda create -n habit python=3.10
   conda activate habit

若出现 ``Proceed ([y]/n)?`` 或类似确认，输入 ``y`` 回车即可。

**中国大陆用户（可选，加速 pip）**：

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

.. code-block:: bash

   conda activate habit

   # Windows：换成你复制的完整路径
   cd "D:\HABIT-main"

   # macOS / Linux：换成你的完整路径
   cd ~/Downloads/HABIT-main

   pip install -r requirements.txt
   pip install -e .

.. warning:: ZIP 解压后可能出现 ``HABIT-main`` 嵌套

   部分解压软件会多出一层 ``HABIT-main/HABIT-main/``。若当前目录里**只有**子文件夹 ``HABIT-main``、没有 ``config`` 和 ``habit``，请再进入内层，重新复制地址栏的**完整路径**后再 ``cd``（例如 ``cd "D:\HABIT-main\HABIT-main"``）。

目录结构示意（ZIP 解压后，文件夹名均为 ``HABIT-main``）：

.. code-block:: text

   HABIT-main/
   ├── config/
   ├── habit/
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
   cd D:\HABIT              # Windows
   # cd ~/Projects/HABIT    # macOS / Linux 示例
   git pull
   pip install -r requirements.txt --upgrade
   pip install -e .

**ZIP 用户**：重新下载并解压（仍会得到 ``HABIT-main``），``cd`` 到该目录（Windows ``cd D:\HABIT-main``；macOS ``cd ~/Downloads/HABIT-main`` 等）后重复 ``pip install -r requirements.txt`` 与 ``pip install -e .``。

卸载：``pip uninstall HABIT -y``（不删除源码与其它依赖）。

常见问题
--------

- 必须使用 **``conda create -n habit python=3.10``**；其它版本易与 PyTorch 等冲突
- 安装失败：``pip install --upgrade pip``；或按 ``requirements.txt`` 逐行 ``pip install``
- 网络超时：配置上文清华镜像；仍失败请 `提交 Issue <https://github.com/lichao312214129/HABIT/issues>`_ 或邮件 **lichao19870617@163.com**

下一步
------

:doc:`quickstart_zh` — 下载 demo 并跑通完整流程。
