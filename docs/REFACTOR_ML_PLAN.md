# 机器学习模块重构计划（医生易懂版）

> 这份文档是给你今天下午直接照着做的。  
> 目标：**不改变你现在的使用方式**（命令行和配置文件照旧），只把内部结构改得更清晰、更稳定。

---

## 1. 先看懂 4 个词（尽量白话）


| 词          | 白话解释                               |
| ---------- | ---------------------------------- |
| `Workflow` | 总流程（像一张检查流程单）                      |
| `Callback` | “流程跑到某一步自动触发的小动作”，例如自动存模型、自动画图     |
| `Runner`   | 执行器：只负责“把模型训练和评估算出来”               |
| `Result`   | 结果包：把本次训练的关键结果统一装在一起，后面要存盘、画图都从这里读 |


你这次的核心改动一句话：

**把“自动触发的小动作（Callback）”改成“我在代码里明确写顺序执行”。**

---

## 2. 为什么要改（用临床思维类比）

现在的状态像这样：

- 主流程在看病（训练模型）
- 旁边有几个“自动助手”（Callback）在你不注意时做事（保存、出报告、画图）
- 一旦助手之间互相依赖，就容易出“顺序问题”

改完后像这样：

- 主流程只做诊断计算（训练/预测）
- 算完拿到一份完整结果单（`RunResult`）
- 再按你指定顺序做：`保存模型 -> 写报告 -> 画图`

好处：

- 更容易排错
- 更容易测试
- 以后加新功能，不容易牵一发而动全身

---

## 3. 本次重构“做什么 / 不做什么”

### 做什么（本次）

- 新增一个结果对象（`RunResult`）
- 新增执行器（`HoldoutRunner`，后续可加 `KFoldRunner`）
- 把旧的 `MachineLearningWorkflow` 改成“薄壳”：内部调用执行器
- 把“存模型、写报告、画图”改成**显式调用**（不用 callback 触发）

### 不做什么（本次）

- 不改 `MLConfig` 格式
- 不改 CLI 用法
- 不重写模型算法本身（如 RF、XGBoost）
- 不改指标公式（AUC、F1 等）
- 不大改 `feature_selectors/` 和 `models/` 细节

---

## 4. 今天要完成的范围

> 今天按顺序完成 **Step 0 ~ Step 6**。  
> 不做时间估计，做完一个勾一个。

---

## 5. 实操步骤（一步一步）

## Step 0：先留“基线结果”

目的：后面好比较“重构前后结果是否一致”。

- 建议新建分支：`refactor/ml-simplify`
- 用你现有 demo 配置跑一次 holdout
- 保存输出目录，命名为 `baseline_holdout`

验收标准：

- baseline 目录里有你平时会产出的 csv/json/模型文件
- 现有测试能跑过（至少跑 ML 相关测试）

---

## Step 1：新增结果对象

新增目录：`habit/core/machine_learning/core/`

建议先只加 2 个文件：

- `plan.py`：放本次运行的固定信息（配置、输出路径、随机种子）
- `results.py`：放结果对象（每个模型结果、整次运行结果）

先不要动旧代码逻辑。

验收标准：

- 新文件能被 import
- 不影响任何现有功能

---

## Step 2：先做 Holdout 执行器

新增文件：`habit/core/machine_learning/runners/holdout.py`

这个执行器只做三件事：

1. 读数据、切分 train/test
2. 训练每个模型并计算指标
3. 返回 `RunResult`

**注意：这里不做存盘、不画图。**

验收标准：

- `HoldoutRunner.run()` 能跑通
- 返回结果里能拿到每个模型的指标和预测结果

---

## Step 3：把“输出动作”拆出来

新增 3 个文件（放在 `reporting/` 下）：

- `model_store.py`：负责保存模型
- `report_writer.py`：负责写 `summary.csv`、`results.json`、`all_prediction_results.csv`
- `plot_composer.py`：负责画图

每个文件都接收同一个输入：`RunResult`。

验收标准：

- 给它一个 `RunResult`，能独立生成文件
- 不依赖 callback

---

## Step 4：改旧入口，但保证外部不变

改文件：

- `habit/core/machine_learning/workflows/holdout_workflow.py`

做法：

- 类名和对外方法名保持不变（`MachineLearningWorkflow.run()`）
- 内部改成：
  1. 调 `HoldoutRunner.run()` 拿结果
  2. 再按顺序调用：`ModelStore -> ReportWriter -> PlotComposer`

这样外部用户（包括你自己的脚本）无需改调用方式。

验收标准：

- 旧命令还能跑
- 输出文件类型和数量与 `baseline_holdout` 基本一致

---

## 6. 今天继续完成（原“明天可选”改为今日执行）

### Step 5：把 `kfold_workflow.py` 改成调用 `KFoldRunner`

- 新增：`habit/core/machine_learning/runners/kfold.py`
- 修改：`habit/core/machine_learning/workflows/kfold_workflow.py`
- 目标：`MachineLearningKFoldWorkflow` 对外用法不变，内部改为 `KFoldRunner.run()`

验收标准：

- KFold 流程能跑通
- 输出文件和重构前结构一致（允许时间戳差异）
- 旧入口不报错

### Step 6：在 callback 模块加“弃用提示”（但不删除）

- 修改：`habit/core/machine_learning/callbacks/__init__.py`
- 加 `DeprecationWarning`，提示改用 `reporting` 模块里的新写法
- callback 文件全部保留，避免旧代码立刻崩溃

验收标准：

- 导入 callback 模块时出现弃用提示
- 现有代码仍能运行，不因删除 callback 而报错

### Step 7：把稳定内容合并到正式文档

- 把最终确认稳定的重构说明，合并到：
  - `docs/source/api/machine_learning.rst`
  - `docs/source/development/architecture.rst`
- 这份临时文档 `docs/REFACTOR_ML_PLAN.md` 仅保留当天执行记录；完成后可删除或压缩为简短备忘

验收标准：

- 正式文档中能看到新结构和新调用方式
- 临时文档不再作为长期维护文档

---

## 7. 你可以直接用的“完成判定”

满足以下 4 条，就算今天成功：

- 旧 CLI 命令不变、能运行
- 与 baseline 对比，核心结果一致（允许时间戳不同）
- 不需要 callback 也能完整输出模型/报表/图
- `test_public_run_entrypoints.py` 通过

---

## 8. 风险点（提前避坑）

1. **不要一开始就改 KFold**
  先把 Holdout 跑通，再复制思路到 KFold。
2. **不要一开始就删 callback 文件**
  先“停止使用”，但保留文件，避免外部代码直接报错。
3. **predict 路径先别动**
  今天先改 train（`fit`）路径，predict 放下一轮。
4. **每一步一个小提交**
  出问题可快速回退，不会整盘重来。

---

## 9. 最简文件改动清单（医生版）

今天优先改这些：

- 新增：`habit/core/machine_learning/core/plan.py`
- 新增：`habit/core/machine_learning/core/results.py`
- 新增：`habit/core/machine_learning/runners/holdout.py`
- 新增：`habit/core/machine_learning/reporting/model_store.py`
- 新增：`habit/core/machine_learning/reporting/report_writer.py`
- 新增：`habit/core/machine_learning/reporting/plot_composer.py`
- 修改：`habit/core/machine_learning/workflows/holdout_workflow.py`
- 新增：`habit/core/machine_learning/runners/kfold.py`
- 修改：`habit/core/machine_learning/workflows/kfold_workflow.py`
- 修改：`habit/core/machine_learning/callbacks/__init__.py`
- 修改：`docs/source/api/machine_learning.rst`
- 修改：`docs/source/development/architecture.rst`

今天先不要碰或少碰：

- `feature_selectors/`
- `models/` 内部算法
- `evaluation/metrics.py`

---

## 10. 一句话记忆版

**训练流程只负责“算结果”；保存、写报告、画图由你明确按顺序调用。**

这就是这次重构的全部核心。

---

## 11. 今天收尾检查清单

- `kfold_workflow.py` 已改为调用 `KFoldRunner`
- callback 模块已加入弃用提示，且未删除旧文件
- 稳定内容已合并进正式文档（API + 架构）
- 临时文档保留为执行记录，不作为长期维护文档

---

## 12. Phase 2：架构深化与命名整理（执行记录）

> 本节是 Phase 1 完成后做的第二轮深化，目标是让数据契约对称、依赖图单向、命名贴合职责。

### 12.1 数据契约（`core/`）

- 新增 `core/dataset.py:DatasetSnapshot`：把训练/测试矩阵从 `RunResult` 拆出。
- 新增 `core/protocols.py:WorkflowResult`：作为 reporting 层接受的最小协议。
- `core/plan.py:WorkflowPlan.__post_init__` 对 `MLConfig` 做 `model_copy(deep=True)`，让 plan 真正成为不可变快照。
- `core/results.py` 重写：
  - `RunResult` 用 `dataset: DatasetSnapshot` 代替 `x_train/x_test/y_*/label_col`，旧字段保留为 `@property` 兼容；`ModelResult` 增加 `train_subject_ids` / `test_subject_ids`。
  - `KFoldRunResult.results: Dict[str, Any]` 不再是不透明字段，被替换为 `models: Dict[str, KFoldModelResult]` + `aggregated: Dict[str, AggregatedModelResult]`，`results` 改为 `@property` 提供旧 dict 视图。
  - 新增 `InferenceResult` 让 predict 路径也吃同一套契约。

### 12.2 执行层（`runners/`）

- 新增 `runners/context.py:RunnerContext`：封装 `data_manager / pipeline_builder / resampler / logger / config`。
- `BaseRunner` 不再持有 `workflow: Any`，改为持有 `context: RunnerContext` + `plan: WorkflowPlan`，依赖图变为单向：`workflow -> context -> runner`。
- `HoldoutRunner` 和 `KFoldRunner` 全部走 context；`KFoldRunner` 的 `_aggregate_results` 拆成纯函数 `_aggregate_models`，不再边写边返回。
- 新增 `runners/inference.py:InferenceRunner`，把 `predict()` 中的核心逻辑搬过来。

### 12.3 报告层（`reporting/`）

- `ModelStore` 同时支持 holdout 与 K-Fold（`save` / `save_kfold_ensembles`）。
- `ReportWriter.write` 用 `isinstance` 路由 holdout / K-Fold / inference 三种结果类型；K-Fold 不再在 workflow 内联 `save_json/save_csv`。
- `PlotComposer.render` 同样按结果类型分派；inference 不出图。

### 12.4 编排层（`workflows/`）

- `BaseWorkflow` 在构造时直接组装 `Resampler` 与 `RunnerContext`；`_train_with_optional_sampling` 保留为兼容包装。
- `holdout_workflow.py`：主类改名为 `HoldoutWorkflow`；`predict()` 走 `InferenceRunner` + `ReportWriter`，不再旁路。
- `kfold_workflow.py`：主类改名为 `KFoldWorkflow`，使用三件套（`ModelStore.save_kfold_ensembles` / `ReportWriter.write` / `PlotComposer.render`），不再内联 IO。

### 12.5 命名整理

| 旧名 | 新名 | 备注 |
| --- | --- | --- |
| `MachineLearningWorkflow` | `HoldoutWorkflow` | 旧名保留为 deprecation 子类 |
| `MachineLearningKFoldWorkflow` | `KFoldWorkflow` | 旧名保留为 deprecation 子类 |
| `BaseRunner.workflow` | `BaseRunner.context` | 反向耦合消除 |
| `BaseWorkflow._train_with_optional_sampling` | `Resampler.fit_with_resampling` | 抽出为 adapter；旧方法保留为兼容包装 |
| `KFoldRunResult.results: Dict[str,Any]` | `KFoldRunResult.models / aggregated`（结构化），`results` 仍可用作 `@property` | — |
| `RunResult.x_train/x_test/y_*/label_col` | `RunResult.dataset` | 旧字段保留为 `@property` |
| `KFoldRunner._aggregate_results`（双重副作用） | `_aggregate_models`（纯函数） | — |

### 12.6 测试与公共 API

- `tests/test_public_run_entrypoints.py`：同时断言新名（`HoldoutWorkflow` / `KFoldWorkflow`）与旧名（deprecation 子类）都存在并实现 `run`。
- `tests/test_ml_kfold_result_contract.py`：新增对 `KFoldModelResult` / `AggregatedModelResult` 的存在性断言；新增对 K-Fold workflow 调用 reporting 三件套的断言。
- `MLConfigurator.create_ml_workflow / create_kfold_workflow` 改为构造新名类。
- `habit.core.machine_learning.__init__` 同时导出新名与旧名。

### 12.7 本轮不做（明确边界）

- `comparison_workflow.py:ModelComparison`（870 行 facade）：与本轮 plan/runner/result 重构正交，下一轮单独处理。
- `feature_selectors/`、`models/` 算法库、`evaluation/metrics.py`、`visualization/plot_manager.py`：均不在本轮深化范围内。
- `to_legacy_results()` / `to_legacy_dict()`：保留为过渡 adapter，等 plotting 与 report 完全升级后再统一移除。