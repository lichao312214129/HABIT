贡献指南
========

感谢你对 HABIT 的兴趣！我们欢迎任何形式的贡献。

如何贡献
----------

报告 Bug
~~~~~~~~

如果你发现了 bug，请在 GitHub 上提交 issue：

1. 搜索现有的 issues，确认问题未被报告
2. 创建新的 issue，包含：
   * 清晰的标题
   * 详细的问题描述
   * 复现步骤
   * 预期行为
   * 实际行为
   * 环境信息（Python 版本、操作系统等）

提交代码
~~~~~~~~

1. Fork 项目
2. 创建你的特性分支 (``git checkout -b feature/AmazingFeature``)
3. 提交你的更改 (``git commit -m 'Add some AmazingFeature'``)
4. 推送到分支 (``git push origin feature/AmazingFeature``)
5. 提交 Pull Request

代码规范
~~~~~~~~

* 遵循 PEP 8 编码规范
* 添加适当的文档字符串
* 为新功能添加测试
* 确保所有测试通过

文档贡献
~~~~~~~~

文档同样欢迎贡献！你可以：

* 修正拼写错误
* 改进现有文档
* 添加新的教程或示例
* 翻译文档到其他语言

开发环境设置
------------

.. code-block:: bash

   # 克隆仓库
   git clone https://github.com/lichao312214129/HABIT.git
   cd habit_project
   
   # 创建虚拟环境
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   
   # 安装开发依赖
   pip install -e ".[dev]"
   
   # 运行测试
   pytest tests/

提交 Pull Request
----------------

提交 PR 前，请确保：

* 代码通过所有测试
* 添加了适当的文档
* 更新了相关的文档
* 遵循代码规范
* PR 描述清晰说明了更改内容

行为准则
----------

* 尊重所有贡献者
* 接受建设性的批评
* 关注对社区最有利的事情
* 对其他社区成员表示同理心

感谢你的贡献！
