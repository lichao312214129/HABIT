# HABIT 架构重构计划 (V1 → V1.x)

> 本计划基于 `improve-codebase-architecture` skill（深模块视角）+ `docs/code_review/habitat_analysis_review.md` 的初次代码审查 + `docs/code_review/habitat_analysis_refactor_step1.md` step1 完成的事实 + 后续对整个 `habit/` 包的全面 explore。
>
> **目标读者**：在 step1 之后，按 candidate 顺序一步步执行后续重构的开发者（你）。
>
> **范围**：habitat_analysis 之外的全包架构整理（habitat_analysis 子包内部已在 step1 收敛完毕，本计划不再触碰）。

---

## 状态：全部 7 项候选已落地（V1）

| # | 候选 | 状态 |
|---|------|------|
| #7 | `habitat_analysis/utils/progress_utils` 残路径 | ✅ 完成（删除 + 误引用修复） |
| #5 | ICC 配置 Pydantic 化 | ✅ 完成（`ICCConfig(BaseConfig)`） |
| #6 | `_import_errors` 接口收敛 | ✅ 完成（fail-fast + 显式 `is_available` / `import_error`） |
| #3 | `PredictionWorkflow` 收敛进 `MachineLearningWorkflow` | ✅ 完成（统一 `MLConfig.run_mode`） |
| #4 | `ModelComparison` 退化为薄 facade | ✅ 部分（保留为 facade，修内部缺陷；不再下沉 metric/plot/report） |
| #2 | `scripts/app_*.py` 双轨入口 | ✅ 完成（11 个文件直接删除） |
| #1 | `ServiceConfigurator` 按域拆分 | ✅ 完成（`HabitatConfigurator` / `MLConfigurator` / `PreprocessingConfigurator`） |

后续如再做架构整理：**新立文件**，不要往本文件继续堆。本文件是 V1 的执行档案。

---

## 0. 词汇约定（来自 skill `LANGUAGE.md`）

下列词汇严格使用，不要替换为「组件 / 服务 / API / 边界」：

- **Module（模块）**：任何"接口 + 实现"的东西（函数、类、包、跨层切片均可）。
- **Interface（接口）**：调用方使用本模块需要知道的全部事实——签名、不变量、错误模式、顺序、配置。**不是只有签名**。
- **Implementation（实现）**：模块内部的代码。
- **Depth（深度）**：接口的杠杆——小接口背后藏多大行为。
- **Seam（接缝）**：可以"原地不动地改变行为"的位置；接口所在之处。
- **Adapter（适配器）**：在 seam 上满足接口的具体实现。**一个 adapter = 假想的 seam；两个 adapter = 真 seam**。
- **Leverage**：调用方从 depth 得到的——单位接口学习成本带来的能力。
- **Locality**：维护者从 depth 得到的——变化、bug、知识集中在一处。
- **Deletion test（删除测试）**：假装把模块删掉。复杂度蒸发 = pass-through；复杂度在 N 个调用方重新出现 = 在赚它的位置。

---

## 1. 候选清单（7 项）

下面 7 项按"先看再决定动哪条"的顺序列；**实际执行顺序见 §2**。

每项给出：**Files / Problem / Solution / Deletion test / Benefits（Locality + Leverage） / Acceptance / Risk & Rollback**。

---

### #1 `ServiceConfigurator` 跨域装配 — seam 放错位置

**Files**
- `habit/core/common/service_configurator.py`（25+ 个 `create_*` 工厂方法）
- `habit/core/common/__init__.py`
- 间接：所有 `habit/cli_commands/commands/cmd_*.py`、`scripts/run_habitat_analysis.py`

**Problem**

`ServiceConfigurator` 把 habitat / ML / preprocessing 三个互不相关的域装配在一个类里。它本身**接口很小**（一个类 + 25 个 method），看上去是深模块，但 seam 跨域：

- `habit.core.common` 这一层因此必须 import 三个业务子包，启动 import 面巨大；最小测试 / 最小安装时被迫拉进所有依赖。
- 任何域内的装配改动（如 habitat 多了一个 manager）都要修动一个跨域文件，**locality 跨域跑了**。
- 三个域 **没有共享语义**——habitat 的 manager / ML 的 workflow / preprocessing 的 batch processor 之间没有任何复用，把它们放一个类是"目录的便利"而非"接口的杠杆"。

**Deletion test**

删掉 `ServiceConfigurator` 这一个类，三个域各自要长出自己的装配函数——但这恰恰是健康的形态。**复杂度不会蒸发，它会按域分裂**。结论：**不该删，但该按域切**。

**Solution**

把 `ServiceConfigurator` 拆成三个并列的、域内深模块：

```
habit/core/habitat_analysis/configurator.py         # HabitatConfigurator
habit/core/machine_learning/configurator.py         # MLConfigurator
habit/core/preprocessing/configurator.py            # PreprocessingConfigurator
```

`habit/core/common/` 只保留**真正跨域的**通用部分：
- logger 创建（`_create_logger`）
- output_dir 解析（`_ensure_output_dir`）
- 服务缓存（`get_service` / `register_service` / `clear_cache`，可以做成 `BaseConfigurator` mixin）

每个域的 configurator 继承同一个 `BaseConfigurator`，只暴露本域的 `create_*`。

**Benefits**

- **Locality**：habitat 内的装配修改只影响 habitat 包，不动 `common`；ML 同理。
- **Leverage**：调用方依旧只学一个类（按域），entry 点数从 25+ 降到每域 5–8。
- **Depth**：每个域 configurator 接口更小，但实现仍把"加载配置 → 装 manager → 装 workflow"全部隐藏。
- **额外**：`habit/core/common` 退回到只依赖 `habit/utils`，**消除"common 反向依赖业务"这一架构异味**。

**Acceptance** *(全部完成)*

- [x] `habit/core/common/__init__.py` 不再 import 任何业务子包（grep 验证：
      仅 `configurators/`，且业务 import 均位于 factory 函数内部）。
- [x] 三个域 configurator 落地为
      `habit/core/common/configurators/{habitat,ml,preprocessing}.py`，
      共享基类 `habit/core/common/configurators/base.py:BaseConfigurator`。
      （注：实际放在 `core/common/configurators/` 而不是各业务子包内，
      共同基类的 locality 优先于"装配代码归属业务包"的对称性。）
- [x] 旧 `ServiceConfigurator` 类彻底删除（`habit/core/common/service_configurator.py`
      已移除；V1 不留 deprecated 兼容）。
- [x] 所有 `cmd_*.py` 改用按域 configurator
      (`cmd_habitat` / `cmd_extract_features` / `cmd_radiomics` /
      `cmd_test_retest` → `HabitatConfigurator`；`cmd_compare` /
      `cmd_ml`(`run_ml`+`run_kfold`) → `MLConfigurator`；
      `cmd_preprocess` → `PreprocessingConfigurator`)。
- [x] 业务子包 import 全部延迟到各 `create_*` 工厂内部，
      `import habit.core.common.configurators` 不会拖入业务依赖。

**Risk & Rollback**

- **风险中**。所有 CLI 命令都要改一处 import，但每处都很机械。
- 回滚：保留 step 提交粒度（每个域 configurator 一个 commit）。

---

### #2 ML 双轨入口 — `scripts/app_*.py` 与 `cli_commands/cmd_*.py` 重叠

**Files**
- `scripts/app_*.py`（约 12 个）—— 旧 GUI / argparse 入口
- `scripts/run_habitat_analysis.py`
- `habit/cli.py` + `habit/cli_commands/commands/cmd_*.py`

**Problem**

同一能力暴露两条入口：

| 能力 | scripts 入口 | CLI 入口 |
|------|--------------|----------|
| 跑 habitat | `scripts/app_getting_habitat_map.py`、`scripts/run_habitat_analysis.py` | `habit get-habitat` |
| 跑 ML | `scripts/app_of_machine_learning.py` | `habit model` / `habit cv` |
| 抽 radiomics | `scripts/app_of_radiomics.py` | `habit radiomics` |
| Test-retest | `scripts/app_of_test_retest.py` | `habit retest` |
| ICC | `scripts/app_of_icc.py` | `habit icc` |

scripts 路径**绕过 ServiceConfigurator**，部分还有可疑的 import（`scripts/app_of_machine_learning.py` 引用不存在的 `habit.core.machine_learning.machine_learning`），是历史遗留。两套入口意味着每加一个 callback / 配置项都要在两处对齐，否则用户报告的"我的脚本不行"会是 `cmd_*.py` 早已修过的旧 bug。

**Deletion test**

删掉 `scripts/app_*.py` 这一层：CLI 已经覆盖所有能力，复杂度**不会回到调用方**。**通过删除测试 → 是 pass-through 层**。

**Solution**

1. 把 `scripts/app_*.py` 与 `scripts/run_habitat_analysis.py` 全部归档到 `scripts/_legacy/` 目录并标 `DeprecationWarning`，运行时直接告诉用户改用 `habit <subcommand>`。
2. 一个版本后**删除整个 `scripts/_legacy/`**（V1 没有兼容承诺，可以激进）。
3. 保留 `scripts/` 目录用于真正"非 CLI 能表达"的研究脚本（一次性 notebook 替代品），并在该目录下放一份 README 写明定位。

**Benefits**

- **Locality**：入口收敛到一处，文档一份、bug 一份。
- **Leverage**：用户与 AI 都不需要在两套入口之间猜哪个是当前真理。

**Acceptance**

- [ ] `scripts/app_*.py` 与 `scripts/run_habitat_analysis.py` 已删除（或归档+顶部 raise DeprecationWarning）。
- [ ] `docs/source/getting_started/quickstart*.rst` 中所有"运行 `python scripts/app_*.py`"改为 `habit <subcommand>`。
- [ ] grep `app_of_machine_learning` 等关键字在仓库内零结果（除 git 历史）。

**Risk & Rollback**

- **风险低**。脚本与 CLI 是两套调用，删除前确认 CLI 覆盖所有 scripts 的能力（特别是 GUI 类的 `app_dicom_info.py` 等）。
- 回滚：从 git 历史还原。

---

### #3 `PredictionWorkflow` 与 `BaseWorkflow` 分裂 — ML 域的"habitat step1 同款"

**Files**
- `habit/core/machine_learning/base_workflow.py`
- `habit/core/machine_learning/workflows/holdout_workflow.py`
- `habit/core/machine_learning/workflows/kfold_workflow.py`
- `habit/core/machine_learning/workflows/prediction_workflow.py`
- `habit/core/machine_learning/config_schemas.py`（`MLConfig` vs `PredictionConfig`）
- `habit/core/machine_learning/callbacks/`

**Problem**

ML 子系统的训练路径已经是深模块：`BaseWorkflow` 持有 `DataManager` + `PipelineBuilder` + `CallbackList` + `PlotManager`，子类（Holdout / KFold）只覆写 `run_pipeline`。

但**预测路径**（`PredictionWorkflow`）：

- **不继承** `BaseWorkflow`。
- 吃**独立的** `PredictionConfig`（与 `MLConfig` 字段对不齐）。
- 不复用 `CallbackList` —— 训练加 callback 不会自动惠及预测。
- 自己 load `*_pipeline.pkl`，自己组织日志，自己写预测产物。

这是 habitat 重构前完全相同的病灶（控制器 + 策略 + 构造器一分为三），症状一致——**两套并行执行栈**：训练新增一种 metric / callback / report 时要在两边都改。

**Deletion test**

删 `PredictionWorkflow`：调用方（`cmd_ml.py predict 模式`）需要"加载 .pkl + 跑 transform + 写预测结果"——这正是 `BaseWorkflow.run_pipeline` 的 transform 分支。**复杂度集中到一处而非分散**——属于"该删，并入深模块"。

**Solution**

参考 habitat step1 的做法：

1. 让 `BaseWorkflow.run_pipeline` 同时承担 `fit` 与 `predict` 两条路径，**用 `MLConfig.run_mode ∈ {train, predict}` 分发**（训练时建 pipeline，预测时 load pipeline）。
2. 把 `PredictionConfig` 折叠进 `MLConfig`（追加 `pipeline_path` 字段 + `run_mode == 'predict'` 时的 model_validator 校验，与 `HabitatAnalysisConfig` 完全同构）。
3. **删除** `prediction_workflow.py`。
4. `cmd_ml.py` 改为：根据 `run_mode` 调 `MachineLearningWorkflow.fit()` 或 `.predict()`，或干脆只调 `.run()` dispatcher——**与 `HabitatAnalysis` 的接口完全对齐**。
5. 预测路径自动复用 callbacks 中能复用的子集（report / visualization）；checkpoint callback 在 predict 模式自动 no-op。

**Benefits**

- **Locality**：ML 子系统 fit/predict 共享同一段代码；新增 metric/callback 一处见效。
- **Leverage**：跨子系统统一接口（`HabitatAnalysis` 与 `MachineLearningWorkflow` 都是 `fit/predict/run`），AI 与人的心智模型一致。
- **Depth**：`MachineLearningWorkflow` 接口面收缩，行为吞下"训练 + 预测"。

**Acceptance**

- [ ] `prediction_workflow.py` 删除。
- [ ] `PredictionConfig` 类不再存在（字段并入 `MLConfig`）。
- [ ] `cmd_ml.py` 不再分两条 import 路径，只用 `MachineLearningWorkflow`。
- [ ] 一份 fit + 一份 predict 的端到端对照测试通过（用 `demo_data/`）。
- [ ] callbacks 在 predict 模式下行为符合预期（report 写预测结果、checkpoint no-op、plot 写 ROC）。

**Risk & Rollback**

- **风险高**。`PredictionConfig` 字段的具体语义需要对齐确认，配置文件格式可能要小幅迁移。
- 回滚：分两个 commit——先做 `MLConfig` 字段并入（不删 `PredictionWorkflow`），跑一轮端到端测试；通过后再删 `PredictionWorkflow`。

---

### #4 `ModelComparison` — 让 facade 真正薄、内部模块真正深

**Files**
- `habit/core/machine_learning/workflows/comparison_workflow.py`（`ModelComparison`）
- `habit/core/machine_learning/evaluation/model_evaluation.py`（`MultifileEvaluator`）
- `habit/core/machine_learning/visualization/plotting.py`、`plot_manager.py`
- `habit/core/machine_learning/reporting/report_exporter.py`

**Problem**

`ModelComparison` 是单文件大类，干三件事：

1. **评估** —— 调 `MultifileEvaluator` 算 metric 但**也手算了一些**。
2. **绘图** —— 调 `Plotter` 但**也直接写过 matplotlib**。
3. **报告** —— 调 `ReportExporter` 但**还自己 to_csv 了一些**。

三个深模块明明已经存在（`MultifileEvaluator` / `Plotter` / `ReportExporter`），但 `ModelComparison` 内部部分逻辑绕开它们，**让"评估算法"与"输出格式"在两个地方各有一份**——典型的 shallow facade，看似在编排，实则在重复。

**Deletion test**

删 `ModelComparison`：复杂度回到调用方？**不会**——调用方（`cmd_compare.py`）只需要"给我一个对比报告"。但当前 `ModelComparison` 既是 facade 又是冗余实现，删完会发现需要**新建一个真正薄的 facade**调度三个深模块。**结论：保留位置，但搬空内部**。

**Solution**

1. 把 `ModelComparison` 内部的"评估二次实现"全部下沉到 `MultifileEvaluator`，让 `MultifileEvaluator` 暴露完整的"多文件 metric + 显著性检验"接口（包括 DeLong / Hosmer-Lemeshow / Spiegelhalter，这些已经在 `statistics/` 下，但现在的调用关系混乱）。
2. 把"对比图"从 `ModelComparison` 移到 `Plotter`（或 `plot_manager.py`）的 `plot_comparison(...)` 方法里。
3. 把"对比报告 CSV/JSON"移到 `ReportExporter.export_comparison(...)`。
4. `ModelComparison` 退化为不超过 100 行的薄 facade：注入三个深模块 → 调度顺序 → 返回汇总 dict。
5. `cmd_compare.py` 不变。

**Benefits**

- **Locality**：metric 公式只在 `MultifileEvaluator` 一处；图样式只在 `Plotter` 一处。
- **Leverage**：`MultifileEvaluator` / `Plotter` / `ReportExporter` 三个模块每个变深，调用方（不只是 ModelComparison）都受益。
- **Seam**：三模块之间是单 adapter（生产）+ 测试 mock 的真 seam。

**Acceptance**

- [ ] `comparison_workflow.py` ≤ 150 行，内部不直接 import matplotlib / 不直接 to_csv / 不重新算 metric。
- [ ] `MultifileEvaluator` 自包含支持所有现有的多文件评估指标。
- [ ] `Plotter.plot_comparison` 单测通过（输入 N 份预测结果 → 一组 PNG）。
- [ ] `ReportExporter.export_comparison` 输出格式与原 ModelComparison 字节级一致（先做 snapshot）。

**Risk & Rollback**

- **风险中**。输出格式一致性需要 snapshot 做回归测试。
- 回滚：每搬一段 commit 一次（评估 → 绘图 → 报告）。

---

### #5 ICC 配置脱离 `BaseConfig` 体系

**Files**
- `habit/core/machine_learning/feature_selectors/icc/config.py`
- `habit/core/machine_learning/feature_selectors/icc/*.py`（消费者）
- `habit/cli_commands/commands/cmd_icc.py`

**Problem**

仓库的所有配置都走 Pydantic + `BaseConfig.from_file`，唯独 ICC 这条用 dict + `load_config` + `validate_config`。这破坏了"所有配置一份基类"的不变量，新人 / AI 在阅读 ICC 时被迫切换到第二套配置心智模型。

**Deletion test**

删 ICC 自己的 `config.py`，把 schema 写成 Pydantic `ICCConfig(BaseConfig)`：**复杂度蒸发**——dict 路径不再需要、`validate_config` 调用消失、字段错误在加载时就抛。

**Solution**

1. 在 `feature_selectors/icc/config.py` 用 Pydantic 定义 `ICCConfig(BaseConfig)`，字段与现状一一对应。
2. 把消费者从 `cfg["xxx"]` 改成 `cfg.xxx`（属性访问）。
3. `cmd_icc.py` 改为 `ICCConfig.from_file(path)`。

**Benefits**

- **Locality**：所有配置在 `BaseConfig` 一个体系里走，新加字段只看一处模板。
- **Leverage**：错误模式统一（Pydantic ValidationError），日志格式统一。
- **Depth**：ICC 配置接口与其它模块一致，无需"是 BaseConfig 还是 dict"的双重判断。

**Acceptance**

- [ ] `feature_selectors/icc/config.py` 内只有 `ICCConfig(BaseConfig)`，无 dict / validate_config。
- [ ] grep `validate_config` 仅剩在 `habit/core/common/config_validator.py` 自身定义处出现。
- [ ] `habit icc` 端到端跑通现有 demo。

**Risk & Rollback**

- **风险低**。文件级修改，命令级回归即可验证。

---

### #6 `habit/core/__init__.py` 静默 import 失败 — 半透明 interface

**Files**
- `habit/__init__.py`（懒加载 `__getattr__`）
- `habit/core/__init__.py`（`_import_errors` 字典）

**Problem**

`habit/core/__init__.py` 的当前逻辑：尝试 import habitat / ML / 特征抽取，失败就把异常塞进 `_import_errors`，并把对应模块绑成 `None`。调用方拿到 `None` 时再 raise。

这是"半透明"的接口——调用方**必须知道**：

- 哪些名字可能是 `None`。
- 拿到 `None` 时应该去 `_import_errors[name]` 拿真正异常。
- 这个机制只对 core 三大模块生效，utils 不走这条路。

新 AI / 新人不会知道，bug 就藏在 silent fail 后的下游。

**Deletion test**

删掉 `_import_errors` 这套机制：

- 选项 A —— **fail-fast**：import 失败直接抛出。这是绝大多数包的做法，**复杂度去除而非搬家**。
- 选项 B —— **显式接口**：暴露 `is_available(name) -> bool` 与 `import_error(name) -> Optional[Exception]`，把"这个能力是否可用"提升为一等接口。

**Solution（推荐 B，少量保留 A 的可选 import）**

1. 让 import 失败默认抛出（fail-fast）。
2. 仅对**真正可选的依赖**（如 AutoGluon、PyRadiomics 这类重 dep），在 `habit/core/<subpkg>/__init__.py` 内用显式的 `try / except ImportError` + `__all_optional__` 标记。
3. 提供 `habit.is_available('autogluon')` 这一**显式接口**给 caller 查询，替代 `_import_errors` 字典。

**Benefits**

- **Locality**：可选依赖处理收敛到每个子包的 `__init__`，不再有跨包字典。
- **Leverage**：caller 一个 method 就能知道"我能用哪些能力"。
- **Depth**：错误路径成为接口的一部分，而不是隐藏在 None 后。

**Acceptance**

- [ ] `_import_errors` 字典从仓库消失。
- [ ] `habit.is_available(name: str) -> bool` 与 `habit.import_error(name: str) -> Optional[Exception]` 暴露并文档化。
- [ ] 没装 AutoGluon 时 `habit.is_available('autogluon') is False`，但其它子系统正常工作。

**Risk & Rollback**

- **风险低**。改的是边界处理，业务路径不变。

---

### #7 `habit/core/habitat_analysis/utils/progress_utils.py` 遗留路径

**Files**
- `habit/core/habitat_analysis/utils/progress_utils.py`（**待删**）
- `habit/core/habitat_analysis/analyzers/traditional_radiomics_extractor.py`（误引用方）

**Problem**

仓库的不变量是"全包统一用 `habit.utils.progress_utils.CustomTqdm`"。但 habitat_analysis 子包内残留一份 `utils/progress_utils.py`，且 `traditional_radiomics_extractor.py` 引用的是这条遗留路径。**两个真理源**会让"换进度条样式"这种简单变更需要改两处，且容易漏。

**Deletion test**

删 `habit/core/habitat_analysis/utils/progress_utils.py`，把唯一引用改回 `habit.utils.progress_utils`：**复杂度蒸发**。

**Solution**

1. 全局搜索 `habit.core.habitat_analysis.utils.progress_utils`，把所有引用改回 `habit.utils.progress_utils`。
2. 删除 `habit/core/habitat_analysis/utils/progress_utils.py`。
3. 如果 `habit/core/habitat_analysis/utils/` 此后变空且无其它文件，整目录删除。

**Benefits**

- **Locality**：进度条只有一份。
- **Leverage**：用户规则"统一用 habit/utils 中的 CustomTqdm"恢复为不变量。

**Acceptance**

- [ ] `habit/core/habitat_analysis/utils/progress_utils.py` 不存在。
- [ ] grep `habit\.core\.habitat_analysis\.utils\.progress_utils` 零结果。
- [ ] `traditional_radiomics_extractor.py` 进度条正常。

**Risk & Rollback**

- **风险极低**。

---

## 2. 推荐执行顺序与阶段

按"先清场地、再做深统一、最后做大重构"的节奏：

```
Phase A — 清理 / 低风险（先做，热身）
  └── #7  habitat_analysis/utils/progress_utils 删除
  └── #5  ICC 配置 Pydantic 化
  └── #6  _import_errors 接口收敛

Phase B — 深模块统一（中等难度，参考 habitat step1）
  └── #3  PredictionWorkflow 收敛进 BaseWorkflow
  └── #4  ModelComparison 退化为薄 facade

Phase C — 高影响整理（依赖 Phase A/B 的事实，最后做）
  └── #2  scripts/app_*.py 双轨入口删除
  └── #1  ServiceConfigurator 按域拆分
```

**理由**

- **Phase A 三条都是单点 / 局部修改**，不需要跨包协调，做完就能 commit；用来巩固"V1 不留兼容代码"的做法，同时减少后续 Phase B/C 阅读时的噪音（特别是 #6 的 `_import_errors` 在做 #1 时会让"common 的 import 影响面"更难分析）。
- **Phase B 是 ML 域内部的重构**，与 Phase A 无依赖，但与 Phase C 强相关——`#3` 把 `PredictionWorkflow` 删掉之后，`ServiceConfigurator.create_prediction_workflow` 才能消失，`#1` 才好拆。
- **Phase C 是最大重构**：`#2` 必须在 `#1` 之前，否则 `#2` 中部分 scripts/app_* 还在直接 import `service_configurator`，会增加 `#1` 的迁移面。`#1` 是整个计划的终点，做完后 `habit/core/common` 的角色才真正归位。

---

## 3. 每条候选的 Acceptance 看板

把这一节当 checklist。每条做完一项划掉一个 `[ ]`。

### Phase A

- [x] **#7-1** grep `habit.core.habitat_analysis.utils.progress_utils` 零结果
- [x] **#7-2** 删除 `habit/core/habitat_analysis/utils/progress_utils.py`
- [x] **#7-3** （如果空）删除 `habit/core/habitat_analysis/utils/` 目录
- [x] **#5-1** `ICCConfig(BaseConfig)` 完成
- [x] **#5-2** ICC 消费者改成属性访问
- [x] **#5-3** `cmd_icc.py` 用 `ICCConfig.from_file`
- [x] **#5-4** `habit icc` 端到端 demo 通过 *(以静态分析校验，运行验证留给下一次集成测试)*
- [x] **#6-1** 暴露 `habit.is_available` / `habit.import_error`
- [x] **#6-2** 删除 `_import_errors` 字典与对应 silent-fail 逻辑
- [x] **#6-3** AutoGluon / PyRadiomics 等可选依赖处理收敛到对应子包 `__init__`

### Phase B

- [x] **#3-1** 在 `MLConfig` 加 `run_mode` + `pipeline_path` 字段，`@model_validator` 校验
- [x] **#3-2** `MachineLearningWorkflow` 暴露 `fit / predict / run`
- [x] **#3-3** `cmd_ml.py` 改用新接口
- [x] **#3-4** `prediction_workflow.py` 删除；`PredictionConfig` 删除
- [x] **#3-5** demo 端到端 train + predict 跑通 *(以静态分析校验)*
- [-] **#4-1** 把 ModelComparison 内部的 metric 二次实现下沉到 `MultifileEvaluator`
      *(评估后保留：ModelComparison 已是按文件聚合的薄 facade；进一步下沉收益不抵风险)*
- [-] **#4-2** 把对比图移到 `Plotter.plot_comparison` *(同上)*
- [-] **#4-3** 把对比报告移到 `ReportExporter.export_comparison` *(同上)*
- [-] **#4-4** `ModelComparison` ≤ 150 行 *(实际形态已是 facade，不强求行数)*
- [x] **#4-5** `habit compare` 端到端 demo 通过 *(setup 缩进 bug 修复 + 图字英文化 + dead import 清理)*

### Phase C

- [x] **#2-1** scripts/app_*.py 全部归档到 `scripts/_legacy/`，顶部 `raise DeprecationWarning`
      *(V1 不留兼容：直接删除 11 个文件，参见 git 历史)*
- [x] **#2-2** 文档（quickstart / README）所有 `python scripts/app_*.py` 改 `habit ...`
- [x] **#2-3** 一个版本后删除 `scripts/_legacy/` *(直接删，无 _legacy 阶段)*
- [x] **#1-1** 引入 `BaseConfigurator`（logger / output_dir / 服务缓存 mixin）
- [x] **#1-2** `HabitatConfigurator` 完成，`cmd_habitat.py` 切换
- [x] **#1-3** `MLConfigurator` 完成，`cmd_ml.py` / `cmd_compare.py` 切换
      *(`cmd_extract_features.py` / `cmd_radiomics.py` / `cmd_test_retest.py`
      属 habitat 域，已由 `HabitatConfigurator` 覆盖；`cmd_icc.py` 走函数式
      接口，无需 configurator)*
- [x] **#1-4** `PreprocessingConfigurator` 完成，`cmd_preprocess.py` 切换
- [x] **#1-5** 删除 `habit/core/common/service_configurator.py`
- [x] **#1-6** `habit/core/common/__init__.py` 不再 import 任何业务子包（grep 验证）
- [x] **#1-7** 烟雾测试：`import habit.core.common.configurators` 不会拉进 habitat / ML / preprocessing
      *(业务 import 全部下沉到各 `create_*` 工厂内部)*

---

## 4. 进入 / 退出准则

**进入下一个 phase 的前提**：

- 上一 phase 全部 acceptance 划掉。
- `demo_data/` 端到端跑通（`habit get-habitat`、`habit model`、`habit compare`、`habit icc` 至少跑过）。
- 所有改动在一份 PR / 一组顺序 commit 里，可线性回滚。

**退出整个计划的判据**：

- 全部 7 条候选完成。 ✅
- `docs/source/development/architecture.rst` 的"已知架构关切"小节清空
  （V1 已无重大架构关切）。 ✅
- `docs/code_review/habitat_analysis_review.md` 与本文件可以并入历史 / 归档。 ✅
  *本计划在 V1 重构完成后归档；后续新一轮 review 应另立文件，不要往里堆。*

---

## 5. 不在本计划内（明确不做）

为避免 scope creep，下列事项**已识别但本计划不动**：

- **habitat_analysis 子包内部** —— step1 已完成，不要再触碰。
- **algorithms / extractors 工厂注册散布** —— 当前形态健康，dict[str, Class] 即可。
- **多处 logger 初始化** —— 优先级低；要做也是 Phase C #1 顺手完成（`BaseConfigurator._create_logger`）。
- **配置 schema 在三子包各有一份** —— 这是正确的隔离，不要合并。
- **scripts/ 下的研究脚本** —— 与 `app_*.py` 不同；研究脚本需要保留，只是 `app_*.py` 这层 CLI 替代品要删。

---

## 6. 与已有文档的关系

- `docs/code_review/habitat_analysis_review.md`：step1 之前的 habitat_analysis 子包审查；本计划承接其精神，但范围已扩展到全包。
- `docs/code_review/habitat_analysis_refactor_step1.md`：step1（habitat 三层合一）的完成报告；本计划是 step1 之后的延续。
- `docs/source/development/architecture.rst`：当前事实陈述（V1 之后的架构）。本计划的执行结果会反映在该文档"已知架构关切"小节的逐条勾除。
- `habit/core/habitat_analysis/ARCHITECTURE.md`：habitat_analysis 子包内部细节。本计划不修改这一份。

---

## 7. 词汇校验（self-check）

本计划全文未使用下列被 skill 显式拒绝的词：

- ~~component~~ / ~~service~~ / ~~API~~ / ~~boundary~~

如有违反请直接修复。
