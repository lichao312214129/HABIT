贡献者工作流
============

本页给出动手贡献所需的实操信息：环境搭建、测试体系、以及三个高频场景的落地步骤——新增一个 CLI 命令、
新增一个配置步骤 Schema、以及 Web GUI 的 bridge 机制。通用的 PR/提交规范见 :doc:`contributing`。

环境搭建
--------

.. code-block:: bash

   conda activate habit          # Python 3.10
   pip install -e ".[dev]"       # 可编辑安装 + 开发依赖
   pytest tests/                 # 运行测试

代码约定
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 约定
     - 说明
   * - 类型注解
     - 函数输入/输出显式标注类型（如 ``x_train: np.ndarray``）。
   * - 注释语言
     - 代码注释用 **英文**，且要详细讲清意图。
   * - 图内文字
     - 所有由代码生成的图 **一律英文**\ （不含文档正文）。
   * - 工具归集
     - 跨子系统工具统一放 ``habit/utils/``；进度条统一用 ``habit/utils/progress_utils.py``。
   * - 代码风格
     - 遵循 PEP 8，公共函数补 docstring。

测试体系
--------

测试位于仓库根 ``tests/``，配置在 ``tests/pytest.ini``。有两类测试：

1. **pytest 单元 / CLI 测试**\ （``test_*.py``）：用 ``click.testing.CliRunner`` 或直接调 API。
2. **可执行 demo 脚本**：如 ``tests/habitat/habitat_two_step_voxel_radiomics_train.py``，指向 ``config/`` 下的
   demo YAML，从仓库根运行，用于端到端冒烟。

按 marker 选择性运行（marker 定义在 ``pytest.ini``）：

.. code-block:: bash

   pytest tests/ -m unit                 # 仅单元测试
   pytest tests/ -m "habitat and not slow"
   pytest tests/ -m cli                  # 仅 CLI 测试

可用 marker：``slow`` / ``integration`` / ``unit`` / ``preprocessing`` / ``habitat`` / ``ml`` / ``utils`` / ``cli``。

demo 数据流
~~~~~~~~~~~

``tests/conftest.py`` 提供 ``project_root`` / ``demo_data_dir`` 等 fixture；示例数据在 ``demo_data/``。
端到端链路示例见 ``tests/integration/``\ （如 ``workflow_preprocess_to_compare.py``，依次跑
preprocess → get-habitat → extract → model → compare）。配置里的路径相对 YAML 文件解析
（见 :doc:`configuration_system` 的 ``PathResolver``）。

场景一：新增一个 CLI 命令
-------------------------

命令的入口与实现是分离的：``habit/cli.py`` 只声明参数并转发，实际逻辑在 ``habit/commands/cmd_*.py``。

**1. 在 ``habit/cli.py`` 声明命令**\ （延迟导入，保持启动快）：

.. code-block:: python

   @cli.command('my-task')
   @config_option()
   def my_task(config):
       """One-line help shown in `habit --help`."""
       from habit.commands.cmd_my_task import run_my_task
       run_my_task(config)

**2. 在 ``habit/commands/cmd_my_task.py`` 实现**\ （参照 ``cmd_preprocess.py`` 的结构）：

.. code-block:: python

   from habit.commands.common import (
       echo_success, exit_with_error, load_config_or_exit,
   )
   from habit.core.my_subsystem.config_schemas import MyConfig
   from habit.core.my_subsystem.run import run_my_task_from_config

   def run_my_task(config_path: str) -> None:
       """Run my task from a config file."""
       config = load_config_or_exit(MyConfig, config_path)
       try:
           run_my_task_from_config(config)
       except Exception as exc:  # noqa: BLE001
           exit_with_error(f"Error: {exc}")
       echo_success("My task completed successfully!")

**3. 核心逻辑** 放到 ``habit/core/.../run.py`` 的 ``run_*_from_config`` 函数中，
保持 "命令层薄、核心层厚"。这样 Python API 用户也能直接调用核心函数。

**4. 补一个 CLI 测试**\ （``tests/.../test_cli_my_task.py``，用 ``CliRunner``）。

场景二：新增一个配置步骤 Schema
-------------------------------

若新增的算法带有可配置 ``params``，应为其定义参数 Schema 并注册，从而获得 **类型校验** 与 **GUI 表单**。

**1. 定义参数模型**\ （``habit/core/schemas/steps/`` 下）：

.. code-block:: python

   from pydantic import BaseModel, Field

   class MyStepParams(BaseModel):
       # Each field: type + default + constraint gives validation + GUI widget.
       n_clusters: int = Field(3, ge=2, description="Number of clusters")
       metric: str = Field("euclidean", description="Distance metric")

**2. 注册到 ``ParamSchemaRegistry``**\ （见 ``schemas/registry.py`` 的初始化逻辑）：

.. code-block:: python

   ParamSchemaRegistry.register("habitat", "my_step", MyStepParams)

**3. 效果**：加载配置时 ``validate_step_params()`` 会自动校验该步骤的 ``params``；
GUI 通过 ``schemas/reflect.py`` 反射出对应表单字段。机制详见 :doc:`configuration_system`。

.. tip::

   由于 ``BaseConfig`` 使用 ``extra='forbid'``，新增字段务必在 Schema 中声明，否则用户配置里出现该字段会报 "未知字段"。

场景三：Web GUI 与 bridge
-------------------------

``habit-gui/`` 是独立于 ``habit`` 包的 Web GUI，通过两条通道复用核心，而 **不重写业务逻辑**：

.. mermaid::

   flowchart LR
     WEB["React UI (habit-gui/web)"] -->|HTTP| API["FastAPI (habit-gui/api)"]
     API -->|subprocess| CLI["habit CLI (habit &lt;cmd&gt; --config)"]
     API -->|bridge worker| BR["habit-gui/bridge"]
     BR -->|import| SCH["habit.core.schemas (reflect)"]

- **执行通道**：GUI 把表单存成 YAML（写到 ``{project}/reports/gui_configs/``），再以 **子进程** 调用
  ``habit`` CLI 运行，行为与命令行完全一致。
- **Schema 通道**：``habit-gui/bridge`` 反射 ``habit.core.schemas`` 的参数模型，把字段描述符喂给前端动态渲染表单。
  因此 **表单字段永远与 CLI 的 YAML Schema 一致**。

这一设计意味着：给核心新增一个组件 + 参数 Schema 后，GUI 表单可自动获得对应输入项，通常无需改前端代码。

.. seealso::

   - 架构全景见 :doc:`architecture`；配置机制见 :doc:`configuration_system`。
   - 扩展点清单见 :doc:`extension_points`；代码模板见 :doc:`../customization/index`。
