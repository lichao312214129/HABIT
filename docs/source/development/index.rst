开发指南
========

本文档面向 HABIT 项目的开发者。

.. toctree::
   :maxdepth: 1
   :caption: 开发文档

   architecture
   contributing

快速开始
--------

1. Fork 项目仓库
2. 安装依赖: ``pip install -e ".[dev]"``
3. 运行测试: ``pytest tests/``
4. 提交 Pull Request

代码规范
--------

- 遵循 PEP 8 编码规范
- 使用类型提示 (Type Hints)
- 添加文档字符串
- 新功能需附带测试

核心模块
--------

- ``habit.core.preprocessing``: 图像预处理
- ``habit.core.habitat_analysis``: 生境分析
- ``habit.core.machine_learning``: 机器学习
- ``habit.utils``: 通用工具

联系方式
--------

有问题请在 GitHub 上提交 Issue。
