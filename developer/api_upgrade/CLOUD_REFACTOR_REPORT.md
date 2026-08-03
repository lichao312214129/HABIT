# HABIT v1.0 云端重构报告（阶段 1–6）

> 本文档汇报云端 agent 在 `cursor/api-first-refactor-2935` 分支上完成的 v1.0 API-first 重构工作，
> 覆盖《07_v1_refactor_plan_and_usage.md》中标记为 ☁️ 的阶段 1–6。
> 设计依据：《06_v1_api_first_architecture.md》；命名依据：《08_naming_decisions.md》（唯一权威）。
>
> 分支：`cursor/api-first-refactor-2935`（基于 `v1.0.0`）
> 报告日期：2026-08-03

---

## 1. 总览

| 阶段 | 内容 | 状态 | 提交 |
|---|---|---|---|
| 1 | L2 契约层 + L1 目录数据源 + 串行后端 | ✅ 完成 | `cfa9b40` |
| — | 命名评审应用（协议重命名 + 契约缺陷修复） | ✅ 完成 | `5cdcb29` |
| 2 | L0 数值核 + L3 领域协议 + registry/spec 三分 | ✅ 完成 | `c5eaf6b` |
| 3 | L5 适配：旧 YAML 翻译、migrate-config、双 schema 校验 | ✅ 完成 | `66f6a39` `292420c` `55a42da` `be43475` `9313e8f` |
| 4 | ProcessPoolBackend + CheckpointStore | ✅ 完成 | `8808fe5` `bad3c4d` |
| 5 | 特征提取与机器学习子系统 | ✅ 完成 | `985ded8` `60eeb9a` `d137157` `12be90a` `7988dab` `a27babc` `a7e7ce0` |
| 6 | 生态适配（sklearn/MONAI/nnU-Net）+ 报告导出 | ✅ 完成 | `a244ec0` `197f383` `ff222a4` `a650bdd` |

**云端测试总结果**：`834 passed, 5 skipped, 62 deselected (slow)`（约 48s）。
62 个 slow 用例与 `tests/integration/` 全部依赖 `demo_data/` 真实影像，按任务约束属于 💻 本地验证范畴（见 §6）。

```bash
# 云端验收命令（在仓库根目录、.venv 环境下）
.venv/bin/python -m pytest tests/ -q --ignore=tests/integration -m "not slow"
# => 834 passed, 5 skipped, 62 deselected, 245 warnings in 47.98s
```

新增包的公开符号通过既有架构契约守护：分层依赖（`test_architecture_contracts.py`）、
公开 API 表面（`tests/api/test_public_api.py`）、打包契约、安装器契约共 242 项全部通过。

---

## 2. 分阶段交付明细

### 阶段 1 · L2 契约层 + L1 目录数据源 + 串行后端（`cfa9b40`）

纯新增地基，不改变 v0.1.x 任何既有行为。

- `habit/contracts/`（L2，八个模块）：
  - `geometry.py`：`Geometry`（shape/spacing/origin/direction 一体）、`GeometryPolicy` 与 `validate_geometry` 几何校验。
  - `image.py`：`ImageVolume` / `MaskVolume` 物化体数据、`ImageRef` 惰性引用协议、`ArrayImageRef` 内存实现。
  - `subject.py`：`Subject`（images/masks 按角色名寻址）与 `Cohort`（`summarize()` 产出 `CohortFingerprint`）。
  - `table.py`：`FeatureTable`——列语义显式（`id_columns` / `feature_columns` / `outcome_column`），
    `feature_matrix()` 只含特征列；`join()` 合并特征族并合并 provenance。
  - `habitat.py`：`VoxelFeatureField` / `Supervoxelization` / `HabitatMap` / `HabitatModel`
    （一等公民工件：`save`/`load`、版本化持久化、`summary()`、`assigner()`）。
  - `provenance.py`：`Provenance` 不可变记录 + `derive()`，为全链路传播打底。
  - `manifest.py`：`RunManifest` / `StudyResult` 骨架（阶段 6 充实报告导出）。
  - `ops.py`：`SubjectOperator` / `CohortOperator` / `DataSource` / `ResultWriter` / `ExecutionBackend` 协议。
- `habit/adapters/directory.py`：`DirectoryDataSource` 读取 HABIT 目录布局；`cohort_from_directory` 便捷入口。
- `habit/execution/`：`SerialBackend` 与执行骨架。
- 测试：`tests/contracts/` 33 项、`tests/adapters/` 5 项。

### 命名评审应用（`5cdcb29`）

按《08_naming_decisions.md》对阶段 1 产物全面改名并修复评审发现的真实契约缺陷
（详见 §4 命名清单）。

### 阶段 2 · 生境分割垂直切片（`c5eaf6b`，☁️ 结构部分）

- `habit/kernels/`（L0，不 import 任何 habit 模块）：`habitat_metrics.py`（MSI/ITH 数值核）、
  `icc.py`、`statistics.py`。
- `habit/domain/protocols.py`：五大领域协议 + `Seedable`（`set_random_state(seed)`）。
- `habit/registry/core.py`：`ComponentRegistry`——`<Registry>.create(name, **params)` /
  `@<Registry>.register("name")`；registry 域名为协议名 snake_case 单数。
- `habit/spec/`：`Spec` / `HabitatSpec` / `RunPolicy` 三分，`fingerprint()` 稳定哈希，
  YAML 同构（`yaml_io.py`）。
- 领域注册表与首个实现：voxel `raw`、supervoxel `slic`、fitter `kmeans`/`gmm`、
  assigner `nearest_centroid`；`SubjectPipeline` 组合垂直切片，`supervoxelizer=None` 选择直接聚类设计。
- 测试：`tests/kernels/` 26 项、`tests/registry/` 10 项、`tests/spec/` 112 项、`tests/domain/`（部分）。

### 阶段 3 · L5 适配：CLI 与旧 YAML 走新核心（`55a42da` `be43475` `9313e8f`，另有两个预备提交）

- `habit/spec/legacy.py`：`LegacyConfigAdapter` 把 v0.1 配置树翻译成 v1 `HabitatSpec` + `RunPolicy`；
  `detect_yaml_version` 双 schema 识别。
- CLI：`migrate-config` 命令、`check-config` 双 schema 校验；CLI 保持薄壳，行为对非编程用户无感。
- 预备提交：`RunPolicy` 执行面扩展（`workers`、`on_subject_failure="continue"/"fail_fast"`、
  超时/重试/断点字段）、KMeans 肘部法 `n_habitats="auto"` 选择、`uv.lock` 可复现云端构建。
- 测试：CLI + legacy 翻译共 109 项。

### 阶段 4 · ProcessPoolBackend 迁移（`8808fe5` `bad3c4d`）

- `habit/execution/process_pool.py`：移植 v0.1 并行机制——`workers` 进程池、
  惰性 `Subject` 的序列化边界（只有路径跨进程，体素不跨进程）、`fail_fast`/`continue` 失败策略、
  超时与优雅关闭。
- `habit/execution/checkpoint.py`：`CheckpointStore` 实现 v0.1 的断点续跑规则，
  记录每受试者成败，供 `RunManifest.subject_outcomes` 汇总。
- 测试：`tests/execution/` 30 项（含多进程实跑，云端容器内通过）。

### 阶段 5 · 特征提取与机器学习子系统（七个提交）

- `985ded8` L0 评估统计核；`d137157` `metric` 域与 L3 统计包装。
- `60eeb9a` 表-ML 协议（`habit/domain/table_protocols.py`）+ `table_preprocessor` 域（8 个预处理器）。
- `12be90a` `feature_selector` 域（12 个选择器）。
- `7988dab` `classifier` 域（14 个分类器；域名用 `classifier` 而非 `model`，与 HabitatModel 区分）。
- `a27babc` `TablePipeline`（版本化持久化，`save`/`load`）。
- `a7e7ce0` 七个 habitat 特征族提取器（`msi`/`ith_score`/`volume`/`each_habitat`/`whole_habitat`/
  `traditional`/`non_radiomics`）与 v1 插件域接线；`habitat_model_fitter` 域承接 `kmeans`/`gmm`
  （修复 v0.1 把它们混在单一 `clustering` 种类下的结构缺陷）。
- 测试：`tests/domain/` 累计 146 项。

### 阶段 6 · 生态适配与报告导出（四个提交）

- `a244ec0` `FileImageRef` 提升为公开 L1 构件（`habit/adapters/image_refs.py`）：
  原本私有于 `DirectoryDataSource` 的惰性文件引用，nnU-Net 数据源需要同样行为，
  故提取为可复用、可子类化的公开类，并纳入公开 API。
- `197f383` `habit/compat` 包（可选依赖、惰性子模块加载，不污染核心）：
  - `compat.sklearn`：`HabitatFeaturesEstimator` 把 `HabitatSpec` 包装成真正的 sklearn
    `BaseEstimator`——`fit` 只在训练队列上学生境定义，`transform` 把后续队列投影到该固定定义，
    交叉验证折永远无法泄漏进生境定义；`n_habitats`/`n_supervoxels`/`random_seed` 以构造参数暴露，
    可直接进 `Pipeline`/`GridSearchCV`。`TableTransformerEstimator`/`TableClassifierEstimator`
    适配表-ML 协议，`FeatureTable` 语义（结局列随行）在 sklearn 管道内保持；
    三个工厂函数 `as_estimator`/`as_transformer`/`as_classifier`。
  - `compat.monai`：`to_monai_dict`/`from_monai_dict` 双向转换（affine↔geometry 精确互转，
    `*_meta_dict` 伴随键、标量路由进 metadata、torch/MetaTensor 鸭子类型转换，零硬依赖）；
    `AsMonaiDict`/`FromMonaiDict`/`AsDictTransform` 三个包装类使 HABIT 个体级算子直接成为
    MONAI transform，可进 `monai.transforms.Compose` 与 torch `DataLoader`。
  - `compat.nnunet`：`NnUNetDataSource` 直读 `imagesTr/labelsTr` + `dataset.json`
    （v2 `channel_names` 与 v1 `modality` 均支持），多标签文件按 `roi_label`
    （整数/标签名/整数并集）在加载时二值化；队列保持全惰性。
  - 配套：`habit/__init__.py` 支持核心子包惰性解析（`habit.compat.sklearn` 式访问保持轻量）；
    `voxel_units` 提升为公开函数（一步式生境设计与 sklearn 适配器复用）。
- `ff222a4` 报告导出：
  - `RunManifest.describe_methods(style)`：从 provenance DAG 渲染**实际执行过**的分析
    （按 provenance 顺序的执行步骤、记录的规格、软件版本、随机种子、被排除受试者），
    `radiology` 开场给软件句、`nature` 收尾给软件句，事实完全一致。
  - `RunManifest.checklist(standard)`：IBSI/CLEAR/METRICS/TRIPOD+AI 四个标准逐条对照，
    机器可取证项给证据，其余诚实地标 `needs_human_answer`，绝不伪造合规。
  - `HabitatSpec.describe_methods(style)`：同名同签名，渲染**计划做**的分析（跑前即可读），
    供预注册与跑前核对 YAML；措辞片段在两个层间有意重复，使 `habit.spec` 不向上依赖。
- `a650bdd` 生态适配契约测试 24 项（全合成数据）：
  sklearn 适配器 10 项、MONAI 转换 9 项、nnU-Net 数据源 5 项。
- 阶段 6 测试合计：`tests/compat/` 24 项 + `tests/spec/` 新增 2 项（`describe_methods` 两种风格
  与错误路径），全部通过。

> 6.5「每个 compat 一个 notebook 示例」按任务约定在云端环境跳过（无可执行界面与影像数据），
> 移交本地阶段 7 处理。

---

## 3. 测试命令与通过数汇总

| 范围 | 命令 | 结果 |
|---|---|---|
| 全部（云端可跑） | `.venv/bin/python -m pytest tests/ -q --ignore=tests/integration -m "not slow"` | **834 passed**, 5 skipped, 62 deselected |
| 架构+API+打包契约 | `pytest tests/test_architecture_contracts.py tests/api tests/test_installer_contracts.py tests/test_packaging_contracts.py` | 242 项全过 |
| L2 契约层 | `pytest tests/contracts tests/adapters` | 38 项全过 |
| L0 核 + registry + spec | `pytest tests/kernels tests/registry tests/spec` | 148 项全过 |
| CLI/YAML 适配 | `pytest tests/test_cli_check_config.py tests/test_cli_migrate_config.py tests/spec/test_legacy.py` | 109 项全过 |
| 执行后端（含多进程） | `pytest tests/execution` | 30 项全过 |
| L3 领域（含表-ML） | `pytest tests/domain` | 146 项全过 |
| 生态适配（阶段 6） | `pytest tests/compat` | 24 项全过 |

5 个 skip 均为环境条件型（可选后端缺失时的既有跳过逻辑），与本次重构无关。

---

## 4. 命名决策应用清单（《08》逐条核对）

| 决策 | 应用情况 |
|---|---|
| `HabitatModelEstimator` → `HabitatModelFitter`，fit 返回新 `HabitatModel` | ✅ 协议与 kmeans/gmm 实现、registry 域名 |
| `HabitatMapper` → `HabitatAssigner`；`HabitatModel.assigner()`；动词 `assign` | ✅ `habitat_assigner` 域、`nearest_centroid` 实现 |
| `Outcome` → `SubjectResult`（`@dataclass(frozen=True)`，`.result()`） | ✅ `habit/contracts/ops.py` |
| `ArtifactSink` → `ResultWriter` | ✅ `habit/contracts/ops.py` |
| `SeedControl` → `Seedable`；`SubjectLevelOp`/`CohortLevelOp` → `SubjectOperator`/`CohortOperator` | ✅ `habit/domain/protocols.py`、`habit/contracts/ops.py` |
| `HabitatModel.describe()` → `.summary()`；`DataSource.cohort()` → `.load()`；`Cohort.fingerprint()` → `.summarize()` | ✅ 全部数据源（directory/nnunet）均为 `.load()` |
| `FeatureTable.label_column` → `outcome_column`；`.features()` → `.feature_matrix()` | ✅ `habit/contracts/table.py` 及全部调用点 |
| 删除所有 `extract/build/map = __call__` 别名 | ✅ 每协议单一 `__call__` |
| 删除 `HabitatFeatureExtractor.name` 属性 | ✅ 注册名只活在 `spec.name` |
| registry 域名 = snake_case 单数协议名；entry point 组 = `habit.<domain>` | ✅ 十域：`voxel_feature_extractor`、`supervoxelizer`、`habitat_model_fitter`、`habitat_assigner`、`habitat_feature_extractor`、`preprocessor`、`table_preprocessor`、`classifier`、`feature_selector`、`metric` |
| `slic` → `supervoxelizer`；`kmeans`/`gmm` → `habitat_model_fitter`（不再混 `clustering`） | ✅ |
| ML 模型 → `classifier` 域 | ✅ 14 个分类器 |
| `<Registry>.create/.register`；顶层 `list_plugins`/`get_plugin_info`/`get_param_schema`/`load_plugins` | ✅ 未发明其它入口 |
| Recipes `two_step_habitat()` 等 | ⏸️ L4 recipes 不在 ☁️ 阶段 1–6 范围（见 §5 偏差说明） |
| 包名保持 `habit.contracts`/`kernels`/`adapters`/`domain`/`execution`/`registry`/`spec`/`recipes`/`compat` | ✅ 未改名；`compat` 本阶段落地 |
| `StudyResult.habitat_model/.pipeline/.features/.habitat_maps/.manifest` | ✅ |
| 运行参数 `workers`、`"continue"/"fail_fast"`、`n_supervoxels`、`n_habitats`、`set_random_state(seed)` | ✅ `RunPolicy` 与 `Seedable` 全程一致 |

---

## 5. 偏差说明

1. **L4 recipes 层未建**：原始简报的 ☁️ 阶段清单为阶段 1–6（契约→协议/垂直切片→CLI/YAML→
   ProcessPool→特征/ML→compat），`habit.recipes`（`two_step_habitat()` 等）不在其中；
   其验收依赖真实影像端到端运行，归 💻 本地后续阶段。公开 API 中尚无 recipes 符号。
2. **6.5 notebook 示例跳过**：云端无可执行界面与影像数据，按任务约定移交本地。
3. **`FileImageRef` 提取为独立模块**：计划文档只说「NnUNetDataSource 需要公开 FileImageRef」，
   实现将其放入新模块 `habit/adapters/image_refs.py` 而非塞进 `directory.py`，
   因为 nnU-Net 数据源与目录布局无关，独立模块避免错误的归属关系；`directory.py` 改为复用。
4. **`voxel_units` 提升为公开**：原私有助手是一步式（直接聚类）设计的必要构件，
   sklearn 适配器与用户的自定义设计都需要它，故按公开函数导出（`habit/domain/pipeline.py`）。
5. **措辞片段有意重复**：`describe_methods` 的组件措辞在 `habit/contracts/manifest.py` 与
   `habit/spec/specs.py` 各存一份（约 60 行）。两个层互不依赖是分层底线，
   spec 层若 import contracts 层会把基础值对象拽进上层依赖。
6. **新增可选依赖**：云端环境为真实验证 MONAI 转换安装了 `torch`(CPU) 与 `monai`
   （测试用 `pytest.importorskip` 守护，缺依赖环境自动跳过，不影响核心套件）。
7. **旧 CLI/公开 API 未破坏**：所有新增均为加法式；既有 834 项测试中包含 v0.1 命令与
   配置长尾的 schema 校验，全部保持绿色。

---

## 6. 💻 本地验证清单（云端无法执行，逐条给出命令）

> 以下各项依赖 `demo_data/`（未入 git）或需要桌面/发布环境，必须在本地完成。
> 在仓库根目录、已安装开发依赖的环境中执行。

1. **阶段 0 golden 基线比对（阶段 2 的数值验收）**：
   ```bash
   pytest tests/integration -m "not slow" -v
   ```
   对三种 `clustering_mode` 验证生境标签图与基线逐体素一致、MSI/ITH 在容差内一致。

2. **62 个 slow 配置长尾（62 配置 × schema 校验 + CLI 实跑）**：
   ```bash
   pytest tests/ -m "slow" -v
   ```
   其中 `test_cli_run_exits_zero[*]` 逐一跑通 `config/` 下全部流水线模板。

3. **端到端 15 个 CLI 命令（demo 数据）**：
   ```bash
   python -m habit run --config config/habitat_analysis/habitat_analysis.yaml
   # 以及 config/ 下其余 14 个模板；对照 tests/QUICKSTART.md 的命令清单
   ```

4. **大队列 ProcessPoolBackend 验收（阶段 4 退出条件）**：
   ```bash
   # 用大队列配置实跑，验证断点续跑：中断后重跑同一命令应跳过已完成受试者
   python -m habit run --config config/habitat_analysis/habitat_analysis.yaml  # workers>1
   pytest tests/execution -m "slow" -v
   ```

5. **影像特征数值验收（阶段 5 退出条件）**：
   ```bash
   # 对照 v0.1.x 输出的 CSV，验证 radiomics/traditional 特征在容差内一致
   pytest tests/machine_learning tests/feature_extraction -v
   ```

6. **MONAI/nnU-Net 真实数据互操作（阶段 6 收尾）**：
   ```bash
   pip install "monai[all]>=1.4" torch --extra-index-url https://download.pytorch.org/whl/cpu
   pytest tests/compat -v
   # 再用一个真实 nnU-Net 数据集验证 NnUNetDataSource 的端到端读取与二值化
   ```

7. **每个 compat 一个 notebook 示例（任务 6.5）**：在本地 Jupyter 环境编写并实跑
   `compat.sklearn`/`compat.monai`/`compat.nnunet` 三个示例 notebook。

8. **阶段 7 文档与发布**：用户文档、API 参考、迁移指南、PyPI/conda-forge 发布、
   冻结 v0.1.x（详见《07》§3 阶段 7）。

---

## 7. 环境说明

- 云端环境基于 `.cursor/environment.json`，`uv.lock` 已提交以保证可复现构建。
- 本阶段额外安装 `torch`(CPU 版) + `monai` 用于真实验证 MONAI 适配（其他云 agent 若需复跑
  `tests/compat` 的 MONAI 用例，建议通过 Cursor 环境配置预置这两个可选依赖；
  缺依赖时相关用例自动 skip，不阻塞其余套件）。
