开发指南
========

本文档面向 HABIT 项目的开发者。

.. toctree::
   :maxdepth: 1
   :caption: 开发文档

   architecture
   module_architecture
   parallel_optimization
   contributing

快速开始
--------

1. Fork 项目仓库
2. 使用 py310 环境（与 :doc:`../getting_started/installation_zh` 一致）::

      conda create -n habit python=3.10 -y
      conda activate habit
      pip install -r requirements.txt
      pip install -e ".[dev]"

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

- ``habit.core.preprocessing`` : 图像预处理
- ``habit.core.habitat_analysis`` : 生境分析
- ``habit.core.machine_learning`` : 机器学习
- ``habit.utils`` : 通用工具

架构与可靠性说明
----------------

- 并发调度优化策略：:doc:`parallel_optimization`
  （Event 驱动唤醒、OOM backoff、自适应超时、poll 限频等）
- 生境批量并行可靠性改造计划：``docs/HABITAT_PARALLEL_RELIABILITY_PLAN.md``
  （GPU worker 槽位、processes 与 GPU 池对齐、同次运行 auto-retry 等）

联系方式
--------

有问题请在 GitHub 上提交 Issue。
