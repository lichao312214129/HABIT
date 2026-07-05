配置系统
========

配置系统是 HABIT 的中枢。理解它，就理解了 "一份 YAML 如何变成一次运行"。本页覆盖：加载与路径解析、Schema 三层结构、参数注册表与校验、领域 Configurator，以及它们之间的关系。

总览：从 YAML 到运行时对象
--------------------------

.. mermaid::

   flowchart TD
     Y["YAML file"] --> LC["load_config()<br/>common/configs/loader.py"]
     LC --> PR["PathResolver<br/>relative paths -> absolute"]
     PR --> FD["BaseConfig.from_dict()<br/>common/configs/base.py"]
     FD --> WF["Workflow schema (Pydantic)<br/>schemas/workflows/*.py"]
     WF --> VP["validate_step_params()<br/>schemas/validation.py"]
     VP --> REG["ParamSchemaRegistry<br/>schemas/registry.py"]
     REG --> SP["Step params model<br/>schemas/steps/*.py"]
     WF --> CFG["Domain Configurator<br/>*/configurator.py"]
     CFG --> OBJ["Runtime object<br/>BatchProcessor / HabitatAnalysis / Workflow"]

分为三块职责：

- **加载层**\ （``common/configs``）：读文件、解析路径、把 dict 灌进 Pydantic 模型。
- **Schema 层**\ （``schemas``）：定义配置结构、校验规则、以及步骤参数的注册与反射。
- **装配层**\ （``configurators``）：把校验后的配置对象变成可执行对象。

加载与路径解析
--------------

入口在 ``habit/core/common/configs/``：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 符号
     - 作用
   * - ``loader.load_config(path, resolve_paths=True)``
     - 读取 YAML/JSON，返回 dict；默认调用 ``PathResolver`` 把相对路径解析为绝对路径。
   * - ``loader.PathResolver``
     - 以配置文件所在目录为基准解析路径（含 Windows/WSL 转换）。这使配置可以写相对路径、随项目移动。
   * - ``base.BaseConfig``
     - 所有配置根模型的基类（继承 Pydantic ``BaseModel``），``model_config`` 设为 ``extra='forbid'``，即 **多写的未知字段会直接报错**。
   * - ``BaseConfig.from_file(path)``
     - 便捷入口：``load_config`` + 路径解析 + ``from_dict``，一步得到强类型配置对象。
   * - ``base.ConfigValidationError``
     - 校验失败时抛出的统一异常，携带出错文件与 Pydantic 错误明细。

.. tip::

   ``extra='forbid'`` 意味着 YAML 里写错字段名（例如把 ``n_clusters`` 拼成 ``n_cluster``）会在加载阶段立刻失败，
   而不是被静默忽略。这是有意为之，用来尽早暴露配置错误。

Schema 三层结构
---------------

``habit/core/schemas/`` 把配置结构拆成三层，各司其职：

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - 层
     - 位置
     - 作用
   * - **Workflows**
     - ``schemas/workflows/*.py``
     - 整份 YAML 的根模型，如 ``PreprocessingConfig`` / ``HabitatAnalysisConfig`` / ``MLConfig``。承载跨字段的整体校验。
   * - **Steps**
     - ``schemas/steps/*.py``
     - 单个步骤 ``params`` 的类型定义，如 ``ResampleParams`` / ``LogisticRegressionParams``。
   * - **Registry**
     - ``schemas/registry.py``
     - ``ParamSchemaRegistry``：把 ``(domain, step_type)`` 映射到对应的 Steps 参数模型。

为什么要分 Workflows 与 Steps？因为很多子系统的配置是 "一个步骤列表"，每个步骤形如 ``{method: xxx, params: {...}}``。
Workflow 模型描述外层骨架，Steps 模型描述每个 ``params`` 的具体字段，二者通过注册表连接。

参数注册表与步骤校验
--------------------

``ParamSchemaRegistry`` 是 Schema 层的枢纽（``schemas/registry.py``）：

.. code-block:: python

   # 注册一个步骤参数模型
   ParamSchemaRegistry.register("preprocessing", "my_step", MyStepParams)

   # 按 (domain, step_type) 取回模型
   model = ParamSchemaRegistry.get("preprocessing", "my_step")

   # 首次使用前触发注册（import schemas 包时自动完成）
   ParamSchemaRegistry.ensure_initialized()

工作流在解析步骤列表时，会调用 ``schemas/validation.py`` 的 ``validate_step_params()``，
按每个步骤的名字从注册表取出对应参数模型并逐一校验：

.. mermaid::

   flowchart LR
     STEP["step: {method, params}"] --> LOOKUP["ParamSchemaRegistry.get(domain, method)"]
     LOOKUP --> MODEL["Params model (schemas/steps)"]
     MODEL --> CHECK["validate params<br/>types / ranges / required"]
     CHECK -->|ok| NEXT["accepted"]
     CHECK -->|fail| ERR["ConfigValidationError"]

注册表同时驱动 GUI 表单：``_wire_factories()`` 会把参数模型挂接到 ``PreprocessorFactory`` / ``ModelFactory`` /
选择器注册表等，供 GUI 反射（见下节）。

领域 Configurator
-----------------

配置校验通过后，由各子系统的 ``configurator.py`` 把配置装配成可执行对象。它们都继承 ``common/configurators/base.py`` 的 ``BaseConfigurator``\ （统一处理日志、输出目录、服务缓存）：

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Configurator
     - 位置
     - 产出对象
   * - ``PreprocessingConfigurator``
     - ``preprocessing/configurator.py``
     - ``BatchProcessor``
   * - ``HabitatConfigurator``
     - ``habitat_analysis/configurator.py``
     - ``HabitatAnalysis`` / ``HabitatMapAnalyzer`` / radiomics / test-retest
   * - ``MLConfigurator``
     - ``machine_learning/configurator.py``
     - ``HoldoutWorkflow`` / ``KFoldWorkflow`` / ``ModelComparison`` 及评估器

三者关系小结
------------

.. code-block:: text

   schemas/workflows/*.py   整份 YAML 结构 + 跨字段校验
   schemas/steps/*.py       单步 params 的类型定义
   ParamSchemaRegistry      把 step 名映射到 steps 参数模型
   validate_step_params()   解析步骤列表时逐步校验 params
   BaseConfig / loader      文件级：load + 路径解析 + from_dict
   Domain Configurator      已校验配置 -> 可执行对象
   运行时 Factory/Registry  按 method 名实例化具体算法

.. note::

   历史兼容：``preprocessing/config_schemas.py`` 等文件只是从 ``schemas/workflows/`` 重新导出（re-export），
   **规范定义在 ``schemas/`` 下**。新增或修改字段请改 ``schemas/``。

GUI Schema 反射
---------------

``schemas/reflect.py`` 与 ``field_reflect.py`` 把 Pydantic 参数模型转成 JSON 字段描述符（类型、默认值、取值范围、分组等），
供 ``habit-gui`` 的表单动态渲染。这样 **CLI 的 YAML 与 GUI 的表单共用同一套 Schema 定义**，字段永远一致。
细节见 :doc:`dev_workflow` 的 "GUI bridge" 一节。

.. seealso::

   - 运行时工厂/注册表清单见 :doc:`extension_points`。
   - 配置字段的用户视角说明见 :doc:`../configuration/index`。
