HABIT 包导入策略指南
========================

V1 导入策略
-----------

HABIT V1 采用 **fail-fast** 导入策略：核心模块（``HabitatAnalysis`` /
``HabitatFeatureExtractor`` / ``Modeling``）的导入失败会**直接抛出 ``ImportError``**，
不会被静默吞掉为 ``None``。

这是设计决定。原因：

1. ``HabitatAnalysis`` / ``Modeling`` 都是 HABIT 的核心能力。它们 import
   失败必然意味着环境损坏或依赖缺失，调用方拿到 ``None`` 后再触发的下游
   报错只会让排错更困难。
2. 让接口"半透明"（有时是类、有时是 ``None``）会迫使每个调用方都做
   ``if foo is not None`` 防御，**locality** 跨调用方分散。

只有 **真正可选的第三方依赖**（V1 当前仅 ``autogluon``）才暴露显式的
查询接口：

.. code-block:: python

   import habit

   if habit.is_available('autogluon'):
       # AutoGluon 装上了，可以用 AutoGluonTabularModel
       ...
   else:
       err = habit.import_error('autogluon')
       print(f"AutoGluon 不可用：{err}")

显式接口
--------

``habit.is_available(name: str) -> bool``
    查询某个**已登记的可选依赖**是否可被 import。

``habit.import_error(name: str) -> Optional[ImportError]``
    返回该可选依赖在最近一次探测时缓存的 ``ImportError``；可用则返回 ``None``。

可选依赖白名单存放在 ``habit._OPTIONAL_DEPENDENCIES``（V1 = ``("autogluon",)``）。
向白名单加项**必须是刻意修改**——这是为了避免新依赖被悄悄当作可选。

调用 ``is_available`` / ``import_error`` 时若传入未登记的名字，会抛
``ValueError``，提示当前白名单内容。

正确的使用模式
--------------

**核心能力使用** ——直接 import，不需要做任何"是否可用"判断：

.. code-block:: python

   import habit

   analysis = habit.HabitatAnalysis(config, ...)
   analysis.fit()

   model = habit.Modeling(config_path)
   model.run()

如果上述 import 失败，你拿到的是真正的 ``ImportError``，traceback 会指向
真正的问题（比如缺 ``SimpleITK``、``pyradiomics``），而不是模糊的 "object
has no attribute"。

**可选能力（如 AutoGluon）** ——先查询：

.. code-block:: python

   import habit

   if habit.is_available('autogluon'):
       from habit.core.machine_learning.models import AutoGluonTabularModel
       model = AutoGluonTabularModel(config)
   else:
       # 退回到默认 ensemble 或提示用户安装
       ...

通用 ImportManager（utils 层）
-----------------------------

如果你在写自己的脚本/工具，需要 **批量、临时** 探测一组依赖（不限于 HABIT
登记的可选项），可以直接用 ``habit.utils.import_utils`` 提供的工具：

.. code-block:: python

   from habit.utils.import_utils import ImportManager, check_dependencies

   manager = ImportManager()
   plt = manager.safe_import('matplotlib.pyplot', alias='plt')
   if plt is None:
       print("matplotlib 不可用：", manager.get_import_errors().get('plt'))

   status = check_dependencies(
       required_modules=['numpy', 'pandas', 'sklearn'],
       optional_modules=['matplotlib', 'seaborn'],
   )
   for name, ok in status.items():
       print('OK' if ok else 'MISS', name)

注意：``ImportManager`` 是 **utils 层** 的通用工具，与上面的
``habit.is_available`` / ``habit.import_error`` **不是同一回事**。前者是
你自己的脚本可以随手用的"safe import 容器"；后者是 HABIT 暴露给用户的
**显式接口**，仅覆盖经过白名单登记的可选依赖。

V0 行为已移除
-------------

V0 曾在 ``habit/core/__init__.py`` 与 ``habit/core/habitat_analysis/__init__.py``
里维护 ``_import_errors`` / ``_available_classes`` 字典并暴露
``get_import_errors`` / ``get_available_classes`` / ``is_class_available``
三个函数。V1 已经移除这套机制：核心 import 直接 fail-fast，可选 dep 走
``is_available`` / ``import_error``。

如果你的旧代码用过下列 API，请按映射改写：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - V0
     - V1
   * - ``habit.get_import_errors()``
     - 直接 ``import habit``，失败会抛真正的 ``ImportError``。
   * - ``habit.get_available_classes()``
     - 不再需要——核心类是必然可用的。
   * - ``habit.is_class_available('HabitatAnalysis')``
     - 不再需要。
   * - ``habit.is_class_available('AutoGluonTabularModel')``
     - 改为 ``habit.is_available('autogluon')``。

总结
----

V1 的导入策略以"接口诚实"为目标：

- **核心模块** —— fail-fast，错误信息清晰。
- **可选依赖** —— 白名单 + 显式 ``is_available`` / ``import_error``。
- **业务无关的临时探测** —— 用 ``habit.utils.import_utils`` 的 ``ImportManager``。

调用方因此可以放心写 ``import habit; habit.HabitatAnalysis(...)``，
不需要再被 "万一是 None" 这种历史遗留迫使写防御代码。
