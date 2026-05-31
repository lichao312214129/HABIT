安装指南
========

本指南面向**首次使用 Python** 的临床与研究人员，从零完成 Miniconda/Anaconda 安装与 HABIT 配置。无需安装 VS Code 等 IDE，在终端中即可完成全部操作。

系统要求
---------

- **Python**: **3.10**（推荐；兼容 PyTorch、AutoGluon 等依赖）
- **操作系统**: Windows / macOS / Linux
- **内存**: 建议 8 GB 及以上（视数据量而定）

一、安装 Miniconda 或 Anaconda
-------------------------------

**Miniconda** 与 **Anaconda** 二选一即可；Miniconda 体积更小，推荐新用户使用。

Miniconda
~~~~~~~~~

1. 打开官方下载页：`Miniconda 官方下载 <https://docs.anaconda.com/miniconda>`_
2. 选择对应系统的安装包并下载。
3. Windows 用户也可直接下载 exe：

   `Miniconda3 Windows x86_64 <https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_

4. 双击安装包，按向导一路 **Next / 继续** 完成安装。

Anaconda（备选）
~~~~~~~~~~~~~~~~

若您更习惯 Anaconda 全家桶，可从 `Anaconda 下载页 <https://www.anaconda.com/download>`_ 下载并安装，后续步骤相同。

打开终端（Windows）
~~~~~~~~~~~~~~~~~~~

安装完成后，从开始菜单打开 **Anaconda Prompt** 或 **Anaconda Powershell Prompt**：

.. code-block:: text

   +--------------------------------------------------+
   |  [开始]  搜索: Anaconda Prompt                    |
   +--------------------------------------------------+
   |  > Anaconda Prompt                               |
   |  > Anaconda Powershell Prompt   <-- 任选其一     |
   +--------------------------------------------------+

macOS / Linux
~~~~~~~~~~~~~

打开系统自带的 **Terminal（终端）**。macOS 首次安装 Miniconda 后，请执行：

.. code-block:: bash

   conda init
   # 关闭并重新打开终端

验证 conda 是否可用
~~~~~~~~~~~~~~~~~~~

命令行前缀出现 ``(base)`` 即表示成功：

.. code-block:: text

   +--------------------------------------------------+
   | Anaconda Prompt                          - x     |
   +--------------------------------------------------+
   | (base) C:\Users\YourName>_                       |
   |       ^^^^                                       |
   |       出现 (base) 表示 conda 已就绪               |
   +--------------------------------------------------+

二、创建 HABIT 专用环境
-----------------------

推荐使用 **Python 3.10**，以便 PyTorch、AutoGluon 等组件正常工作。

.. code-block:: bash

   conda create -n habit python=3.10 -y
   conda activate habit

激活后前缀由 ``(base)`` 变为 ``(habit)``：

.. code-block:: text

   (base) C:\Users\YourName> conda activate habit
   (habit) C:\Users\YourName>_
          ^^^^^^

创建环境时若提示 ``Proceed ([y]/n)?``，输入 ``y`` 并回车。

三、配置 pip 镜像（中国大陆推荐）
---------------------------------

可显著加快依赖下载，并减少超时错误：

.. code-block:: bash

   pip config set global.timeout 6000
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   pip config set global.extra-index-url https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
   pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

四、下载源码并安装 HABIT
------------------------

下载源码
~~~~~~~~

**方式 A：ZIP 下载（推荐普通用户）**

打开 `HABIT GitHub 页面 <https://github.com/lichao312214129/HABIT>`_，点击 **Code → Download ZIP**，或直接访问：

`https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip <https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip>`_

解压到固定路径，例如 ``D:\HABIT`` 或 ``D:\HABIT-main``。

**方式 B：Git 克隆（适合开发者）**

.. code-block:: bash

   git clone --depth 1 --single-branch https://github.com/lichao312214129/HABIT.git
   cd HABIT

进入项目目录
~~~~~~~~~~~~

在 **Anaconda Prompt** 中使用 ``cd`` 切换到包含 ``requirements.txt`` 的目录：

.. code-block:: bash

   cd D:\HABIT
   # 若文件夹名为 HABIT-main: cd D:\HABIT-main
   # Windows 跨盘符: 先输入 D: 回车，再 cd

确认目录结构（模拟界面）：

.. code-block:: text

     D:\HABIT\
     ├── config\
     ├── habit\
     ├── requirements.txt   <-- 必须存在
     └── setup.py

   (habit) D:\HABIT> dir requirements.txt

安装依赖与 HABIT
~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

安装成功时终端末尾会出现类似 ``Successfully installed habit-...`` 的输出。

主要依赖
---------

HABIT 的主要依赖包括：

- SimpleITK: 医学图像处理
- numpy / pandas: 数值与表格处理
- scikit-learn: 机器学习
- pyradiomics: 影像组学特征提取
- click / pyyaml / pydantic: CLI 与配置

完整列表见仓库根目录 ``requirements.txt``。

验证安装
--------

.. code-block:: bash

   habit --version

若输出版本号，说明安装成功。

卸载 HABIT
-----------

.. code-block:: bash

   pip uninstall HABIT -y

**注意**：卸载不会删除源码目录，也不会自动移除其他依赖包。

更新 HABIT
-----------

**Git 用户**：

.. code-block:: bash

   cd HABIT
   git pull
   pip install -r requirements.txt --upgrade
   pip install -e .

**ZIP 用户**：重新下载并解压，在项目目录中再次执行 ``pip install -r requirements.txt`` 与 ``pip install -e .``。

可选依赖
---------

.. code-block:: bash

   pip install antspyx shap matplotlib seaborn

安装问题排查
------------

逐个排查依赖
~~~~~~~~~~~~

若 ``pip install -r requirements.txt`` 报错，可打开 ``requirements.txt`` 逐行安装：

.. code-block:: bash

   pip install SimpleITK
   pip install pyradiomics
   # 针对每一行重复执行

常见问题
~~~~~~~~

- **Python 版本**: 请使用 **3.10** 环境（``conda create -n habit python=3.10``）。
- **pip 版本**: ``pip install --upgrade pip``
- **网络超时**: 确认已配置第三节中的清华镜像。
- **C++ 编译器**: 部分包（如 pyradiomics）在 Windows 上可能需要 Visual Studio Build Tools。

获取支持
~~~~~~~~

- **GitHub Issue**: `提交 Issue <https://github.com/lichao312214129/HABIT/issues>`_
- **电子邮件**: **lichao19870617@163.com**

下一步
------

安装完成后，请阅读 :doc:`quickstart_zh` 下载 demo 数据并跑通完整流程。
