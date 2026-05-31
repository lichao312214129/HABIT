安装指南
========

面向临床与研究人员：安装 Miniconda，创建 **Python 3.10** 环境 ``habit``，安装 HABIT。在 **Anaconda Prompt**（Windows）或系统终端中操作即可。

系统要求
---------

- **Python 3.10**（与项目 ``requirements.txt`` / py310 测试环境一致）
- Windows / macOS / Linux；建议内存 8 GB 及以上

一、安装 Miniconda
------------------

从 `Miniconda 下载 <https://docs.anaconda.com/miniconda>`_ 安装（Windows 可用 `直链 exe <https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_）。也可用 Anaconda，步骤相同。

- **Windows**：开始菜单打开 **Anaconda Prompt** 或 **Anaconda Powershell Prompt**
- **macOS / Linux**：打开终端；首次安装后执行 ``conda init`` 并重启终端

命令行前缀出现 ``(base)`` 即表示 conda 可用。

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

**获取源码**：`下载 ZIP <https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip>`_ 解压，或 ``git clone --depth 1 https://github.com/lichao312214129/HABIT.git``。进入含 ``requirements.txt`` 的目录后安装：

.. code-block:: bash

   cd D:\HABIT
   pip install -r requirements.txt
   pip install -e .

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

.. code-block:: bash

   conda activate habit
   cd D:\HABIT
   git pull
   pip install -r requirements.txt --upgrade
   pip install -e .

ZIP 用户：重新下载解压后重复 ``pip install -r requirements.txt`` 与 ``pip install -e .``。卸载：``pip uninstall HABIT -y``（不删除源码与其它依赖）。

常见问题
--------

- 必须使用 **``conda create -n habit python=3.10``**；其它版本易与 PyTorch 等冲突
- 安装失败：``pip install --upgrade pip``；或按 ``requirements.txt`` 逐行 ``pip install``
- 网络超时：配置上文清华镜像；仍失败请 `提交 Issue <https://github.com/lichao312214129/HABIT/issues>`_ 或邮件 **lichao19870617@163.com**

下一步
------

:doc:`quickstart_zh` — 下载 demo 并跑通完整流程。
