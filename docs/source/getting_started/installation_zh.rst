安装指南
========

系统要求
---------

- **Python**: 3.8 或更高版本。
- **注意**: 基础功能（影像处理、生境分析、传统机器学习）完全支持 Python 3.8。
- **AutoGluon 特别说明**: 如果您计划使用 **AutoGluon** 进行自动机器学习建模，该模块要求 **Python 3.10** 环境。您可以先在 Python 3.8 环境下完成特征提取，然后创建一个新的 Python 3.10 环境来专门运行 AutoGluon。

主要依赖
---------

HABIT 的主要依赖包括：

- SimpleITK: 医学图像处理
- numpy: 数值计算
- pandas: 数据处理
- scikit-learn: 机器学习
- pyradiomics: 影像组学特征提取
- click: 命令行接口
- pyyaml: 配置文件解析
- pydantic: 配置验证

从源码安装
-----------

.. code-block:: bash

   git clone https://github.com/your-repo/habit_project.git
   cd habit_project
   pip install -e .

安装依赖
---------

.. code-block:: bash

   pip install -r requirements.txt

或者使用开发模式安装（推荐用于开发）：

.. code-block:: bash

   pip install -e .

验证安装
--------

验证安装是否成功：

.. code-block:: bash

   habit --version

如果安装成功，您应该看到版本号输出。

可选依赖
---------

某些功能需要额外的依赖包：

- **antspyx**: 用于高级图像配准（可选）
- **shap**: 用于模型解释（可选）
- **matplotlib**: 用于可视化（推荐）
- **seaborn**: 用于高级可视化（推荐）

如果需要这些功能，可以单独安装：

.. code-block:: bash

   pip install antspyx shap matplotlib seaborn

安装问题排查
------------

如果在安装依赖（``pip install -r requirements.txt``）时遇到错误，您可以尝试以下步骤：

1. **逐个排查依赖**
   
   有时某个特定的包可能因为系统环境原因无法安装。您可以打开 ``requirements.txt`` 文件，尝试逐行手动安装，以找出具体是哪个包出了问题：

   .. code-block:: bash

      # 例如：
      pip install SimpleITK
      pip install pyradiomics
      # ... 针对文件中的每一行执行

2. **常见问题检查**

   - **Python 版本**: 确保使用 Python 3.8 或更高版本。
   - **pip 版本**: 建议升级到最新版本 (``pip install --upgrade pip``)。
   - **C++ 构建工具**: 某些包（如 pyradiomics）可能需要系统安装 C++ 编译器。
   - **网络问题**: 如果下载速度慢或超时，可以使用国内镜像源：

     .. code-block:: bash

        pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

3. **获取支持**

   如果问题依然无法解决，请通过以下方式联系我们，并提供报错截图：

   - **GitHub Issue**: `提交一个新的 Issue <https://github.com/lichao312214129/HABIT/issues>`_
   - **电子邮件**: 发送邮件至 **lichao19870617@163.com**
