# Habitat Analysis 架构重构报告（Step 1）

> Skill: `engineering/improve-codebase-architecture`
> Scope: `habit/core/habitat_analysis/`
> 日期: 2026-04-30
> 关联文档: [habitat_analysis_review.md](./habitat_analysis_review.md) §4.1 §4.2 §2.1（C1/C2）§2.2（M4）

---

## 0. 一句话总结

把 **HabitatAnalysis（控制器） → ClusteringStrategy（策略子类树） → pipeline_builder（mode 分支）** 三层拍扁为一个 deep module `HabitatAnalysis`，对外提供 `fit() / predict() / run()` 三个语义清晰的入口，`clustering_mode` 分支只在一处。同时消化 review 列出的 3 个 Critical/必修：C1（反射注入 manager）、C2（predict 路径触发 None 消费）、M4（fit vs fit_transform 不等价）。

---

## 1. 重构前后

### 1.1 重构前（4 处 mode 分支）

```
HabitatAnalysis.run
   └── get_strategy(mode)                       # 分支 #1：strategy 类
         └── OneStep/TwoStep/DirectPoolingStrategy(analysis).run(...)
              ├── _run_train_mode → build_habitat_pipeline(...)  # 分支 #2：builder 内部 if-mode
              ├── _run_predict_mode → HabitatPipeline.load + _update_pipeline_references（反射）
              ├── _post_process_results       # 分支 #3：仅 OneStep 实现
              └── _save_results               # 分支 #4：仅 OneStep 实现

build_habitat_pipeline
   ├── if mode == 'two_step': _build_two_step_pipeline
   ├── elif mode == 'one_step': _build_one_step_pipeline
   └── elif mode == 'direct_pooling': _build_pooling_pipeline
```

外部 caller（`cmd_habitat.py` / `scripts/run_habitat_analysis.py`）在 predict 模式下**绕开** `HabitatAnalysis.run()`，自行 `get_strategy(...).run(load_from=...)` —— 因为 `HabitatAnalysis.run()` 不支持 `load_from`。这是 friction 的物理证据。

### 1.2 重构后（1 处 mode 分支）

```
HabitatAnalysis
   ├── fit(subjects, save_results_csv) -> DataFrame
   │     └── _build_pipeline → fit_transform → save pkl → finalise
   ├── predict(pipeline_path, subjects, save_results_csv) -> DataFrame
   │     └── HabitatPipeline.load → _inject_managers_into_pipeline
   │         → plot_curves=False → transform → finalise
   ├── run(subjects, save_results_csv, load_from=None)  # backward-compat alias
   │     └── 派发到 fit() 或 predict()
   ├── _build_pipeline()                                # 唯一 mode 分支
   │     └── _PIPELINE_RECIPES[mode](config, fm, cm, rm)
   ├── _PIPELINE_RECIPES = {'two_step': _build_two_step_steps,
   │                        'one_step': _build_one_step_steps,
   │                        'direct_pooling': _build_pooling_steps}
   ├── _post_process_results(df)        # 内部按 mode 微调
   └── _save_results(df, mode)          # 内部按 mode 微调
```

---

## 2. 改动文件清单

| 文件 | 性质 | 摘要 |
|---|---|---|
| `habit/core/habitat_analysis/habitat_analysis.py` | **重写** | 吸收 strategy 树 + pipeline_builder 全部逻辑；新增 `fit()` / `predict()`；保留 `run()` alias；显式 manager 白名单；3 个 mode recipe（模块级私有函数）。 |
| `habit/core/habitat_analysis/strategies/base_strategy.py` | **改为薄壳** | 仅保留 `BaseClusteringStrategy` 类作为 deprecated wrapper，`__init__` 触发 `DeprecationWarning`，`run()` 委托给 `HabitatAnalysis.run()`。re-export `_canonical_csv_column_order`。 |
| `habit/core/habitat_analysis/strategies/{one,two,direct_pooling}_strategy.py` | **改为空子类** | 全部空壳（只有 docstring + `pass`），用于保留 `isinstance(s, OneStepStrategy)` 这类 caller 的语义。 |
| `habit/core/habitat_analysis/strategies/__init__.py` | **保留 alias** | `STRATEGY_REGISTRY` / `get_strategy()` 保留并加 `DeprecationWarning`。 |
| `habit/core/habitat_analysis/pipelines/pipeline_builder.py` | **改为薄 wrapper** | `build_habitat_pipeline()` 加 `DeprecationWarning`，函数内部 lazy import `_PIPELINE_RECIPES` 后转调（避免与 `pipelines/__init__.py` 的循环引用）。 |
| `habit/cli_commands/commands/cmd_habitat.py` | **更新 caller** | 删除 `from habit.core.habitat_analysis.strategies import get_strategy`；train 走 `analysis.fit()`，predict 走 `analysis.predict(pipeline_path=...)`。 |
| `scripts/run_habitat_analysis.py` | **更新 caller** | 同上：用 `analysis.fit()` / `analysis.predict()` 替代手工 strategy 装配。 |

未改动的关键依赖：
- `pipelines/base_pipeline.py`（`HabitatPipeline.fit/transform/save/load` 完全不动；pkl 文件向后兼容）
- 三个 manager（`FeatureManager` / `ClusteringManager` / `ResultManager`）
- 11 个 pipeline step（`pipelines/steps/*.py`）
- `algorithms/` 全部
- `extractors/` 全部
- `analyzers/` 全部
- `ServiceConfigurator.create_habitat_analysis(config)` 接口签名不变

---

## 3. 三个公开行为入口

### 3.1 `fit(subjects=None, save_results_csv=None) -> pd.DataFrame`

训练并持久化。流程：build pipeline → `fit_transform` → 保存 `<out_dir>/habitat_pipeline.pkl` → post-process → 保存 `habitats.csv` 和 habitat 图。

### 3.2 `predict(pipeline_path, subjects=None, save_results_csv=None) -> pd.DataFrame`

加载已训练的 pipeline 做 transform。流程：`HabitatPipeline.load(...)` → `_inject_managers_into_pipeline(...)` → 强制 `plot_curves=False` → `transform` → finalise。

### 3.3 `run(subjects=None, save_results_csv=None, load_from=None) -> pd.DataFrame`

Backward-compat 派发：
1. `load_from` 显式给定 → `predict(load_from, ...)`
2. `config.run_mode == 'predict'` 且 `config.pipeline_path` 给定 → `predict(config.pipeline_path, ...)`
3. 否则 → `fit(...)`

---

## 4. 消化的 review 问题

| review 项 | 状态 | 落实方式 |
|---|---|---|
| **C1** `_update_pipeline_references` 反射注入会误覆盖未来 `*_manager` 属性 | ✅ 修复 | 改为 class-level 显式白名单 `_PIPELINE_MANAGER_ATTRS = ('feature_manager', 'clustering_manager', 'result_manager')`。新增 manager 必须显式加入此白名单。 |
| **C2** predict 模式下 `selection_methods=None` 仍可能被 `plot_habitat_scores` 消费 → `TypeError` | ✅ 修复 | `HabitatAnalysis.predict()` 在 `transform()` 前显式 `self.pipeline.config.plot_curves = False`，确保 `PopulationClusteringStep.transform()` 内部 `if self.config.plot_curves and self.habitat_scores_ is not None` 短路。 |
| **M4** `HabitatPipeline.fit` 与 `fit_transform` 两条不等价路径（fit 用 fit→transform，fit_transform 用 fit_transform） | 🟡 部分修复 | 新 `HabitatAnalysis.fit()` 始终走 `pipeline.fit_transform(X)` 一条语义路径，避免 `pipeline.fit()` 那条易错分支。`base_pipeline.py` 内的双实现保留以防外部直接使用，但主流程不再触达 `fit()` 那条路径。后续可独立 PR 把 `HabitatPipeline.fit` / `fit_transform` 合一。 |
| **§4.1** Strategy / PipelineBuilder 双重 mode 分支 | ✅ 解决 | 唯一驻留点 `_PIPELINE_RECIPES`。 |
| **§4.2** `HabitatAnalysis` 是 pass-through shallow module | ✅ 解决 | 升格为 deep module，承担 train/predict 编排、mode 分支、pipeline 构造、引用同步、保存。 |

未在本次 step 1 处理（需独立 PR）：
- M1 / M2 / M3 / M5（散点必修）
- §3 的可读性改进（隐式契约 → TypedDict、文档漂移、`HabitatsSegmention` 拼写）
- §6 性能（O(n²) correlation filter、重复 fit、可视化默认开启）
- §7 测试覆盖（这次没新写测试，下一步独立 PR）
- 候选 #2 / #3 / #4 等其他 deepening opportunity

---

## 5. Caller 迁移指南

### 5.1 训练

**之前**:

```python
configurator = ServiceConfigurator(config, logger=logger)
analysis = configurator.create_habitat_analysis()
strategy_cls = get_strategy(config.HabitatsSegmention.clustering_mode)
strategy = strategy_cls(analysis)
results = strategy.run(subjects=subjects, save_results_csv=True)
```

**之后**:

```python
configurator = ServiceConfigurator(config, logger=logger)
analysis = configurator.create_habitat_analysis()
results = analysis.fit(subjects=subjects, save_results_csv=True)
```

### 5.2 预测

**之前**:

```python
strategy_cls = get_strategy(config.HabitatsSegmention.clustering_mode)
strategy = strategy_cls(analysis)
results = strategy.run(subjects=subjects, save_results_csv=True, load_from=str(pipeline_path))
```

**之后**:

```python
results = analysis.predict(
    pipeline_path=str(pipeline_path),
    subjects=subjects,
    save_results_csv=True,
)
```

### 5.3 仍可用的旧写法（带 DeprecationWarning）

- `from habit.core.habitat_analysis.strategies import get_strategy` —— 仍可用，但触发 warning
- `BaseClusteringStrategy` / `OneStepStrategy` 等 —— 仍可用，构造时触发 warning，`run(load_from=...)` 内部委托
- `from habit.core.habitat_analysis.pipelines import build_habitat_pipeline` —— 仍可用，调用时触发 warning，内部转调新 recipe

---

## 6. 验证

### 6.1 已完成

- ✅ Lint：`ReadLints` 在所有改动文件上为 0 error
- ✅ 静态语法：`python -m py_compile` 在 9 个新增/修改文件上全部通过
- ✅ 静态扫描：再无任何 `*.py` 在 import 已删除的内部成员（`_run_train_mode` / `_select_pipeline_config` / `_update_pipeline_references` 等）
- ✅ Import 路径：`pipeline_builder.py` 改为函数内 lazy import，避免与 `pipelines/__init__.py` 的循环引用

### 6.2 未跑（环境无 numpy/sklearn 等运行时依赖）

- ❌ 端到端：用合成 4×4×4 nrrd × 3 个 mode × {fit, predict} 的 e2e 测试还**没写**
- ❌ 现有 `tests/test_cluster_selection.py` 没在本机跑（算法层未改，预期不受影响）

> 建议：下一个独立 PR 加一份 `tests/integration/test_habitat_analysis_e2e.py`，覆盖 3 个 mode 的 fit + predict 回路。这也将是整个 `habitat_analysis` 包**第一个**端到端测试。

---

## 7. 公开 API 兼容性

| 公开符号 | 状态 | 说明 |
|---|---|---|
| `HabitatAnalysis(config, fm, cm, rm, logger)` | 不变 | 构造签名一致 |
| `HabitatAnalysis.run(subjects, save_results_csv, load_from=None)` | 增强 | 新增 `load_from` 参数；老调用（不传 `load_from`）行为等价 |
| `HabitatAnalysis.fit(...)` / `predict(...)` | 新增 | 推荐入口 |
| `HabitatAnalysis.results_df` / `images_paths` / `mask_paths` / `supervoxel2habitat_clustering` | 不变 | 仍是 property forwarding |
| `ServiceConfigurator.create_habitat_analysis(config)` | 不变 | |
| `habitat_pipeline.pkl` 文件格式 | 不变 | `HabitatPipeline.save/load` 完全未改 |
| `from habit.core.habitat_analysis.strategies import get_strategy / STRATEGY_REGISTRY` | 兼容 | 加 `DeprecationWarning` |
| `from habit.core.habitat_analysis.pipelines import build_habitat_pipeline` | 兼容 | 加 `DeprecationWarning` |
| `BaseClusteringStrategy` / `OneStepStrategy` / `TwoStepStrategy` / `DirectPoolingStrategy` | 兼容 | 类仍存在；构造时触发 `DeprecationWarning`；`run()` 内部委托 |

**已训练的旧 pkl 文件**：完全可以被新代码 `predict` 加载使用（`HabitatPipeline.save/load` 没动；step 内的属性引用通过 `_inject_managers_into_pipeline` 重新注入）。

---

## 8. 已知风险 / 后续 TODO

### 8.1 风险

1. `pipelines/__init__.py` 仍会触发 `pipeline_builder.py` 的导入。`pipeline_builder.py` 函数体内 lazy import `..habitat_analysis._PIPELINE_RECIPES` —— 如果 `habitat_analysis.py` 自身的 import 链里出现 `from ..pipelines.pipeline_builder import build_habitat_pipeline`（目前没有），就会再次循环。**风险源已排除，但需注意未来不要再加这种 import**。
2. predict 模式下 `_inject_managers_into_pipeline` 把当前进程的 `clustering_manager` / `result_manager` 注入加载的 pipeline。若用户用一个**完全新**的 config（`run_mode != 'predict'`、且 `FeatureConstruction` 不为 None）去 `predict()`，行为与之前一致 —— 但这是边界用例，本次未改变其语义。
3. `_save_results` 内的 `mode='train'` / `'predict'` 标记目前只用于日志。如果未来 result 保存差异化（如 predict 不写 supervoxel 图），扩展点已经备好。

### 8.2 推荐的后续小 PR（按 review §8 的 ~100 行/PR 准则）

| # | PR 标题 | 估算行数 | 性质 |
|---|---|---|---|
| step1.1 | Tests: add e2e test for habitat_analysis on synthetic nrrd（3 mode × fit/predict） | ~150 | Test |
| step1.2 | Fix: collapse `HabitatPipeline.fit` and `fit_transform` to a single semantic path（review M4 完整收尾） | ~60 | Refactor |
| step1.3 | Cleanup: remove `try/except ImportError` shims in `habit/core/habitat_analysis/__init__.py`（review M5） | ~60 | Cleanup |
| step1.4 | Refactor: introduce `SubjectPayload` TypedDict for pipeline step IO（review §3.1，候选 #4） | ~120 | Refactor |
| step2 | Refactor: lift "select-k orchestration" out of `BaseClustering` into its own deep module（候选 #2） | ~250 | 大 Refactor |
| step3 | Cleanup: collapse the three clustering registration seams（候选 #3） | ~80 | Cleanup |

### 8.3 一个版本周期后

- 删除 `habit/core/habitat_analysis/strategies/` 整个子包
- 删除 `habit/core/habitat_analysis/pipelines/pipeline_builder.py` 的 deprecated wrapper（recipe builder 留在 `habitat_analysis.py`）
- 从 `__init__.py` 公共 API 中移除 `build_habitat_pipeline`

---

## 9. Skill 流程交叉引用

按 [improve-codebase-architecture skill](../../.cursor/rules/skills/engineering/improve-codebase-architecture/SKILL.md) 流程：

| 阶段 | 产出 |
|---|---|
| Explore | 复用 `habitat_analysis_review.md` + 一次 explore subagent 走读 algorithms/extractors/analyzers，列出 8 个 deepening opportunity |
| Present candidates | 8 个候选按优先级排序，附 deletion test |
| Frame problem space | 列约束 / 依赖类别（全 in-process / local-substitutable）/ caller 物理证据 |
| Spawn 3 sub-agents | 极简 / 高灵活 / 默认极简三种 interface |
| Compare & recommend | 推荐 Hybrid = A 骨架 + C 显式命名 |
| Grilling decisions | 1A / 2A（mode 分支唯一在 `_build_pipeline`）/ 4A（alias 一个版本周期） |
| Execute | 本次 step 1 |

ADR 目录暂未创建（用户选择跳过此步）；如未来重启 ADR 实践，可把本节"决策"汇总为 `docs/adr/0001-collapse-habitat-strategy.md`。
