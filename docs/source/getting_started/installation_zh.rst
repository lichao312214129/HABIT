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

推荐安装步骤（使用 Conda）
--------------------------

为了确保环境的稳定性，我们强烈建议使用 **Miniconda** 或 **Anaconda** 创建独立的虚拟环境。

1. **创建虚拟环境**：

   .. code-block:: bash

      # 推荐使用 Python 3.8。如果需要 AutoGluon，请使用 3.10
      conda create -n habit python=3.8 -y

2. **激活环境**：

   .. code-block:: bash

      conda activate habit

3. **从源码安装**：

   .. code-block:: bash

      git clone https://github.com/lichao312214129/HABIT.git
      cd HABIT
      pip install -r requirements.txt
      pip install -e .

验证安装
--------

验证安装是否成功：

.. code-block:: bash

   habit --version

如果安装成功，您应该看到版本号输出。

卸载 HABIT
-----------

如果需要卸载 HABIT 包，可以使用以下方法：

1. **查看已安装的包名**：

   .. code-block:: bash

      # 查看所有已安装的包
      pip list | grep -i habit

      # 或者查看包的详细信息
      pip show HABIT

2. **卸载包**：

   .. code-block:: bash

      # 卸载 HABIT 包（使用 -y 参数自动确认，避免交互提示）
      pip uninstall HABIT -y

3. **验证卸载**：

   .. code-block:: bash

      # 检查是否已卸载
      pip show HABIT

      # 如果包已卸载，上述命令会提示 "Package(s) not found"

**注意事项**：

- 卸载包不会删除源代码目录，只会移除 Python 环境中的安装链接。
- 卸载包不会自动卸载依赖包，如果需要清理所有依赖，需要手动处理。
- 如果使用 Conda 环境，建议在对应的环境中执行卸载命令。

更新 HABIT
-----------

如果 HABIT 包有新版本发布，您可以通过以下方法更新到最新版本：

**方法 1：从 Git 仓库更新（推荐）**

如果您是通过 `git clone` 安装的，进入项目目录并拉取最新代码：

.. code-block:: bash

   # 进入 HABIT 项目目录
   cd HABIT

   # 拉取最新代码
   git pull

   # 如果依赖有更新，重新安装依赖
   pip install -r requirements.txt --upgrade

   # 重新安装包（确保安装是最新的）
   pip install -e .

**方法 2：重新克隆仓库**

如果遇到合并冲突或想完全重新安装：

.. code-block:: bash

   # 备份您的配置文件（如果有自定义配置）
   # cp -r HABIT/config my_config_backup

   # 删除旧目录
   # rm -rf HABIT

   # 重新克隆
   git clone https://github.com/lichao312214129/HABIT.git
   cd HABIT
   pip install -r requirements.txt
   pip install -e .

**方法 3：仅更新依赖包**

如果只是依赖包有更新，而代码没有变化：

.. code-block:: bash

   # 更新所有依赖到最新版本
   pip install -r requirements.txt --upgrade

**注意事项**：

- 使用 `git pull` 更新时，如果本地有未提交的修改，可能会遇到冲突。建议先提交或暂存本地修改。
- 如果更新后遇到问题，可以查看 `CHANGELOG.md` 或 GitHub Releases 了解版本变更。
- 更新后建议运行 `habit --version` 验证安装是否成功。

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
