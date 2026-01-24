安装指南
========

系统要求
---------

HABIT 需要 Python 3.8 或更高版本。

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

如果在安装过程中遇到问题，请检查以下几点：

1. **Python 版本**: 确保使用 Python 3.8 或更高版本
2. **pip 版本**: 建议使用最新版本的 pip
3. **权限问题**: 如果遇到权限问题，可以使用 `--user` 参数
4. **网络问题**: 如果网络不稳定，可以使用国内镜像源

使用国内镜像源安装：

.. code-block:: bash

   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
