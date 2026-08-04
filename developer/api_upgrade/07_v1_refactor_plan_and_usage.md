# 07 — HABIT v1.0 重构执行计划 与 重构后使用手册

> **本文与 06 的关系**
> [`06_v1_api_first_architecture.md`](./06_v1_api_first_architecture.md) 回答"**架构应该长什么样**"（分层、协议、数据契约、设计取舍）。
> 本文回答两件事：
> - **第一部分**：怎么一步步把现在的代码改成那个样子（阶段、任务、验收、回滚）。
> - **第二部分**：改完之后，别的用户和开发者**具体怎么用** HABIT 的 CLI 与 API（逐模块、逐功能）。
>
> 阅读顺序建议：先 06 建立心智模型，再读本文第一部分排期，第二部分可作为交付时的用户文档草稿。

---

## 第 0 部分 · 前置状态：许可证迁移（已完成）

重构的第一前提是"别人可以合法地把 HABIT 装进自己的东西里"。原先的自定义非商业许可证与 06 的目标 G1（嵌入生态）直接冲突，已完成迁移。

| 项 | 迁移前 | 迁移后 |
|---|---|---|
| `LICENSE` | HABIT Software License（非商业、需书面授权） | Apache License 2.0 官方全文 |
| 源码文件头 | 自定义许可头 | 标准 Apache-2.0 短头 + 版权行，覆盖 409 个 `.py` 文件（不含 `installer/vendor` 的第三方件） |
| `NOTICE` | 无 | 新增。声明 HABIT 版权，并署名两处第三方来源 |
| `CITATION.cff` | 无 | 新增。GitHub 可识别的结构化引用元数据 |
| `setup.py` | `license` 与 classifier 指向自定义许可 | `Apache-2.0` + `License :: OSI Approved :: Apache Software License` |
| `pyproject.toml` | 未声明 license 文件 | `[tool.setuptools] license-files = ["LICENSE", "NOTICE"]` |
| `MANIFEST.in` | 未显式包含 | 显式 `include LICENSE / NOTICE / CITATION.cff` |
| `README.md` / `README_en.md` | 非商业条款说明 | Apache-2.0 说明 + 引用请求（非许可条件） |

上游署名（Apache-2.0 §4 要求）：

| 文件 | 上游 | 上游许可 |
|---|---|---|
| `habit/core/machine_learning/statistics/delong_test.py` | Netflix VMAF | BSD-2-Clause-Patent |
| `habit/core/habitat_analysis/clustering_features/torchradiomics/` | lyhyl/pytorchradiomics | MIT |

**对使用者的实际含义**：学术与商业均可自由使用、修改、再分发、闭源集成，唯一义务是保留版权与许可声明并随分发附带 `NOTICE`。引用 HABIT 从"许可证强制"变为"社区规范"，因此 v1.0 必须在**产物里自动生成可引用的方法学描述**（见第 10.8 节），用工程手段替代法律手段。

这同时关闭了 06 §15 的开放问题 1；其余开放问题的决策记录：

| 开放问题 | 决策 |
|---|---|
| 2. `HabitatModel` 分发形态 | v1.0 先做**自描述的本地文件**（`.habitatmodel`，见 10.5），保证脱离训练目录可用；模型注册中心留到 v1.1 视社区需求再评估 |
| 3. v0.1.x 维护期限 | v1.0 稳定后停止维护 v0.1.x，仅保留 tag 供复现历史结果 |
| 4. 报告规范的对外形态 | 中间路线：先随 HABIT 发布一份建议清单（第 10.8 节的 checklist 导出），若被采用再推动正式共识 |

---

# 第一部分 · 完整重构计划

## 1. 总体策略

### 1.1 分支与发布

| 分支 / tag | 角色 |
|---|---|
| `main` | 保持 v0.1.x 可用，只接受缺陷修复；阶段 0 的 golden 基线在此产出 |
| `v1.0.0` | 重构主线。允许破坏性改动 |
| `v0.1.x` tag | v1.0 正式发布时打 tag 冻结，供复现旧结果 |

预发布节奏：`1.0.0a1`（阶段 2 完成，仅生境分割）→ `1.0.0b1`（阶段 5 完成，全流程）→ `1.0.0rc1`（阶段 6 完成）→ `1.0.0`。

### 1.2 四条不可协商的红线

1. **数值不漂移**：每个阶段的产物必须对齐阶段 0 的 golden 基线。生境标签图逐体素一致；浮点特征在显式记录的容差内一致。
2. **CLI 用户无感**：15 个子命令的名称与选项不变，旧 YAML 继续可跑。
3. **抽象要能用领域语言解释**：无法用"生境分析里的什么概念"说清楚存在理由的抽象一律不要。这是防止过度设计的唯一判据。这条同样适用于命名——**已有名字就沿用，不另造同义词**。
4. **单个体是原子调用**：任何个体级算子都必须能以 `op(subject)` 直接调用，不得要求先构造队列、后端或配置。队列与并行是可选外层。违反这条就等于关掉 MONAI / PyTorch 互操作，也等于让开发者无法单独调试出问题的那一例。

### 1.3 单阶段的完成定义（DoD）

一个阶段只有同时满足下列条件才算完成，否则不进入下一阶段：

- [ ] 新增代码有单元测试，覆盖率不低于该目录重构前的水平
- [ ] `tests/test_architecture_contracts.py` 的分层依赖规则通过
- [ ] 对应的 golden 基线比对脚本通过
- [ ] `import habit` 的冷启动时间不劣化超过 10%
- [ ] 公开符号变更同步进 `habit/api/registry.py`，`tests/api/test_public_api.py` 通过
- [ ] 该阶段涉及的 CLI 命令跑通 `config/` 下对应模板
- [ ] 该阶段新增的每个个体级算子，都有一个"只处理单个 `Subject`、不构造后端"的测试

---

## 2. 重构后的目标包结构

```
habit/
├── __init__.py              # 稳定扁平 API 门面（惰性导入）
├── contracts/               # L2 领域数据契约
│   ├── geometry.py          #   Geometry
│   ├── image.py             #   ImageVolume / MaskVolume / ImageRef
│   ├── subject.py           #   Subject / Cohort
│   ├── habitat.py           #   VoxelFeatureField / Supervoxelization / HabitatMap / HabitatModel
│   ├── table.py             #   FeatureTable
│   ├── provenance.py        #   Provenance
│   └── ops.py               #   SubjectLevelOp / CohortLevelOp / Outcome
├── kernels/                 # L0 纯计算，无 IO 无状态
│   ├── clustering/          #   kmeans / gmm / slic 的数值核
│   ├── voxel_features/      #   raw / kinetic / local_entropy / radiomics 核
│   └── habitat_metrics/     #   MSI / ITH / 空间指标公式（原样迁入，不动数值）
├── adapters/                # L1 唯一允许触碰文件系统的层
│   ├── directory.py         #   DirectoryDataSource（迁移自 io_utils）
│   ├── dataframe.py         #   DataFrameDataSource
│   ├── memory.py            #   InMemoryDataSource
│   ├── nnunet.py            #   NnUNetDataSource
│   └── sinks.py             #   ArtifactSink 实现
├── kernels/                 # L0 纯数值：无 IO、无状态、不 import 任何 habit 模块
│   ├── habitat_metrics.py   #   MSI / ITH / 体积占比等公式
│   ├── feature_transforms.py #  两个预处理域共用的 fit_*/apply_* 内核
│   ├── icc.py
│   └── statistics.py
├── domain/                  # L3 八个领域协议 + 内置实现
│   ├── protocols.py
│   ├── voxel_features/
│   ├── feature_preprocessing/   #   聚类输入预处理（体素/超体素矩阵）
│   ├── supervoxel/
│   ├── supervoxel_features/
│   ├── habitat_model/
│   ├── assignment/
│   ├── habitat_features/
│   └── table_preprocessing/     #   建模表预处理（一行一受试者）
├── execution/               # 执行后端（与算法正交）
│   ├── backends.py          #   SerialBackend / ProcessPoolBackend
│   └── checkpoint.py        #   CheckpointStore
├── registry/                # 组件注册表基类与自省
│   ├── core.py              #   ComponentRegistry 基类：register / create / available
│   └── entrypoints.py       #   habit.<域名> entry point 加载（load_plugins）
├── spec/                    # Spec / RunPolicy / YAML 双向同构
│   ├── specs.py
│   ├── policy.py
│   ├── yaml_io.py
│   └── legacy.py            #   LegacyConfigAdapter：v0 YAML → (Spec, DataSource, RunPolicy)
├── recipes/                 # L4 标准研究配方 + RunManifest + 报告导出
│   ├── habitat.py           #   two_step / one_step / direct_pooling
│   ├── features.py
│   ├── modeling.py
│   ├── manifest.py
│   └── report.py            #   方法学描述与报告清单导出
├── compat/                  # 生态适配（可选依赖）
│   ├── sklearn.py
│   ├── monai.py
│   └── nnunet.py
├── cli.py + commands/       # L5，仅解析与装配
└── utils/                   # 统一工具（进度条、日志、IO 基元）
```

**与现有代码的搬迁映射**见 06 §4.2。核心原则：**数值代码原样迁移，只搬位置不改公式**；改的是它周围的编排、配置读取与 IO。

---

## 3. 阶段计划

> **执行位置约定**：每个阶段标 ☁️（云端可做）或 💻（本地必做）。云端拿不到影像数据（`demo_data` 未入 git），凡涉及真实影像端到端、数值基线比对、CLI 实跑的，都必须在本地。结构性改造、契约测试、csv 驱动的 ML 流程、文档与原型，云端可做。`.cursor/environment.json` 已预置云端环境。

### 阶段 0 · golden 基线固化（💻 本地，在 `main` 上做）

**为什么先做**：没有基线，后面任何"重构后结果不一样"都无法判断是 bug 还是预期。

| 项 | 内容 |
|---|---|
| 任务 | 1) 新建 `tests/golden/` 与 `scripts/make_golden_baseline.py`；2) 用 `demo_data/preprocessed/`（2 受试者 × 3 模态）跑通 `config/habitat/` 下 two_step / one_step / direct_pooling 三种模式；3) 用 `demo_data/ml_data/` 跑 `config/machine_learning/config_machine_learning_kfold_demo.yaml`；4) 固化产物指纹 |
| 锁定内容 | 生境标签图（逐体素 hash）、`msi_features.csv`、`ith_scores.csv`、`habitat_basic_features.csv`、ML 的 AUC / 最佳阈值 / 各折指标 |
| 关键动作 | **显式锁定受试者遍历顺序与所有随机种子**。群体级聚类对二者敏感，不锁死基线没有意义 |
| 容差 | 标签图：逐体素严格一致。浮点特征：`rtol=1e-6`，写进基线元数据 |
| 环境 | `py310`，同时记录 numpy / SimpleITK / scikit-learn / PyRadiomics 版本指纹 |
| 验收 | 同一环境重复跑两次，基线比对脚本报告零差异 |
| 注意 | `demo_data/` 未纳入 git，**本阶段必须在本地跑**，不能交给云端 agent |

---

### 阶段 1 · L2 契约层 + L1 目录数据源 + 串行后端（☁️ 云端可做）

**目标**：把地基钉死。此阶段不改变任何现有行为，纯新增。

| 任务 | 说明 |
|---|---|
| 1.1 建 `habit/contracts/` | 从 `developer/api_upgrade/prototype/contracts.py` 提升为正式实现。`Geometry` / `ImageVolume` / `MaskVolume` 复用现有 `habit/api/image.py` |
| 1.2 实现 `ImageRef` 惰性加载 | 关键设计点。`Subject.images["T1"]` 返回引用，`.load()` 才读盘。需同时验证：`pickle` 可序列化、跨进程传递轻量、可被第三方替换（PACS / zarr / torch tensor） |
| 1.3 `Subject` / `Cohort` | 不可变；`Cohort` 可迭代 / 切片 / `filter`；`subject_id` 唯一性校验 |
| 1.4 `Provenance` | 含 `derive()`，为阶段 2 起的全链路传播打底 |
| 1.5 `DirectoryDataSource` | 迁移 `utils/io_utils.get_image_and_mask_paths` 的目录约定解析，产出 `Cohort` |
| 1.6 `SerialBackend` + `Cohort.map()` | `ExecutionBackend` 最简实现；`Cohort.map(op)` 默认走串行后端，**用户不构造后端也能跑通全流程**。`SubjectLevelOp` 定为"带 `spec`/`cache_key` 的可调用对象"，不引入第二个方法名 |
| 1.7 扩展架构契约测试 | 在 `tests/test_architecture_contracts.py` 加分层依赖断言：L0 不导入 L1+，L2 不导入 L3+，只有 L1 与 L4 Sink 可触碰文件系统 |

**验收**：契约测试全绿；`DirectoryDataSource` 从 `demo_data/preprocessed/` 正确构出 2 例 `Cohort`；`import habit` 仍轻量。
**回滚**：本阶段纯新增，直接 revert 即可，`main` 行为不受影响。

---

### 阶段 2 · 生境分割垂直切片（**关键阶段**，☁️ 结构 + 💻 数值验收）

**这是整套设计的证明题**。如果生境分割能在新架构下对着基线逐值复现，设计成立；不成立就停下重新评估，不要继续往下摊。

**分工**：五个协议落地、`HabitatAnalysis` 拆解、`HabitatModel` 实现、registry 统一、Spec 三分（任务 2.1–2.7）是结构性工作，**云端可做**；但"逐体素一致 / MSI / ITH 容差内"这条验收依赖影像数据，**必须在本地对阶段 0 基线跑**。云端交付代码与契约测试，本地跑数值验收。

| 任务 | 说明 |
|---|---|
| 2.1 落地五个领域协议 | `VoxelFeatureExtractor` / `Supervoxelizer` / `HabitatModelEstimator` / `HabitatMapper` / `HabitatFeatureExtractor`，见 06 §6。统一 `__call__` + 领域动词别名（`extract = __call__`）；`HabitatMapper` 的模型构造期注入，使其成为单参可调用并让"未 fit 先 predict"不可表达 |
| 2.1b `SubjectPipeline` 与 `SeedControl` | 个体级链条合成单个可调用对象（HABIT 版 Compose，保留异构类型）；随机组件统一 `set_random_state(seed)`，取代各自为政的 `random_state` |
| 2.2 拆解 `HabitatAnalysis` | 现有 god object 按"数据 / 算法 / 执行 / 输出"四个关注点切开，数值逻辑原样搬进 `kernels/` |
| 2.3 实现 `HabitatModel` | 含 `describe()` / `save()` / `load()`。**用版本化自描述格式，不用裸 pickle**；跨版本要么可读，要么给出明确不兼容提示 |
| 2.4 统一 registry | 八个现有 registry 收编到同一个 `ComponentRegistry` 基类（后缀统一为 `Factory`，`FeatureExtractorRegistry` / `SelectorRegistry` / `MetricRegistry` 保留旧名为别名）；为聚类、特征表预处理、特征选择补上缺失的插件域，使现有的 `list_plugins` / `get_plugin_info` / `get_param_schema` 覆盖全部八类 |
| 2.5 `two_step` 配方 | 在 `recipes/habitat.py` 装配；`one_step` 与 `direct_pooling` 随后 |
| 2.6 消灭 `_PIPELINE_RECIPES` | 三种模式变为五协议的不同装配方式，第四种模式由用户自行组装，不需改 HABIT 源码 |
| 2.7 Spec 三分 | `Spec`（算法参数）/ `DataSource`（数据在哪）/ `RunPolicy`（怎么跑），见 06 §8 |

> **落地后与本表的偏差**（以实现为准，第二部分已按实现修订）：协议是**八个**而非五个——多出 `SupervoxelFeatureExtractor`（划分与描述是两个正交轴）以及 `SubjectFeaturePreprocessor` / `CohortFeaturePreprocessor`（原计划错误地并入 `table_preprocessor` 域，而那个协议处理的是建模表不是聚类矩阵）；`HabitatModelEstimator` 定名为 `HabitatModelFitter`，`HabitatMapper` 定名为 `HabitatAssigner`（`model.assigner()`）；注册表后缀统一为 **`Registry`** 而非 `Factory`；领域动词别名（`extract = __call__`）**未实现**且已否决（类体别名会在子类覆写 `__call__` 后静默分叉）；配方层（任务 2.5）尚未落地。

**验收**：三种 `clustering_mode` 的生境标签图与阶段 0 基线**逐体素一致**；MSI / ITH 在容差内一致。
**退出条件**：若逐值一致做不到且定位不出原因，暂停后续阶段，重新评估设计。
**回滚**：`v1.0.0` 分支在阶段 1 末尾打 tag `v1-phase1`，可回退。

---

### 阶段 3 · L5 适配：CLI 与旧 YAML 走新核心（☁️ 结构 + 💻 CLI 实跑验收）

**目标**：兑现"CLI 用户无感"。

| 任务 | 说明 |
|---|---|
| 3.1 `LegacyConfigAdapter` | 冻结 v0 schema，翻译为 `(Spec, DataSource, RunPolicy)`。这是长尾工作量所在 |
| 3.2 `habit/commands/` 改调 L4 | 命令层只做解析与装配，不含业务逻辑 |
| 3.3 新增 `habit migrate-config` | v0 YAML → v1 YAML；**不迁移也能继续跑** |
| 3.4 `check-config` 支持双 schema | 自动识别 v0 / v1 |
| 3.5 用 `tests/test_all_configs.py` 兜长尾 | 现有 62 个流水线配置逐个覆盖翻译层，包括隐式默认值。`config/` 共 74 个 YAML，其余 12 个是清单与 PyRadiomics 参数预设，不走翻译层 |

**验收（"无感"的可检验定义）**：
- 15 个命令退出码与 v0.1 一致
- **科学结果文件逐值一致**（特征 CSV / habitat map / 模型指标）
- 用户需要修改的 YAML 字段集合不变

允许改善的部分：输出目录结构、日志格式、进度条呈现。

---

### 阶段 4 · `ProcessPoolBackend` 迁移（☁️ 结构 + 💻 大队列验收）

**目标**：把现有在并行、超时、OOM 退避、断点续跑上的工程投入完整搬过来，且**算法代码零改动**。

从 config 移出、收进后端的字段：
`processes`、`cap_processes_to_gpu_pool`、`individual_subject_timeout_sec`、`individual_subject_graceful_shutdown_sec`、`individual_subject_spawn_timeout_sec`、`on_subject_failure`、`oom_backoff`、`oom_reduce_workers_by`、`resume`

| 任务 | 说明 |
|---|---|
| 4.1 `ProcessPoolBackend` | 迁移超时 / 优雅关闭 / 单例失败隔离 / OOM 退避降并发 |
| 4.2 `CheckpointStore` | 断点续跑作为后端的正交关注点，不再散落在算法里 |
| 4.3 惰性 `Subject` × 多进程 | 阶段 1 的设计在这里真正受检验：序列化边界与缓存边界 |
| 4.4 GPU 池协调 | 保留 `cap_processes_to_gpu_pool` 语义 |

**验收**：大队列跑通；`--resume` 行为与 v0.1 一致；人为 kill 后续跑产物与不中断跑一致。

---

### 阶段 5 · 特征提取与机器学习子系统迁移（☁️ 大部分可做，💻 影像特征验收）

| 任务 | 说明 |
|---|---|
| 5.1 生境特征提取 | 6 个内置插件（`traditional` / `non_radiomics` / `whole_habitat` / `each_habitat` / `msi` / `ith_score`）接入 `HabitatFeatureExtractor` 协议 |
| 5.2 传统影像组学 | `habit radiomics` 路径迁移 |
| 5.3 `FeatureTable` 贯通 | 特征表预处理（8 个方法）、特征选择（12 个选择器）、建模（14 个模型）统一在 `FeatureTable` 上操作，明确区分 ID 列 / 特征列 / 标签列 |
| 5.4 训练/预测一致性 | 预处理与选择的**拟合态**随模型持久化，杜绝预测期重新拟合导致的泄漏 |
| 5.5 评估与统计 | 9 个指标、DeLong、ICC、test-retest 迁移 |

**验收**：`config/machine_learning/` 与 `config/feature_extraction/` 下模板产物对齐基线。

---

### 阶段 6 · 生态适配与报告导出（☁️ 云端可做，MONAI/nnU-Net 需装对应可选依赖）

| 任务 | 说明 |
|---|---|
| 6.1 `compat.sklearn` | 领域协议包装成 `BaseEstimator`，可进 `Pipeline` / `GridSearchCV` |
| 6.2 `compat.monai` | `Subject` ↔ MONAI dict 双向转换；HABIT 算子可作 MONAI transform |
| 6.3 `compat.nnunet` | `NnUNetDataSource` 直读 `imagesTr/labelsTr` + `dataset.json` |
| 6.4 报告导出 | 由 `Provenance` 自动生成方法学段落与 IBSI / CLEAR / TRIPOD+AI 条目对照清单 |
| 6.5 可运行示例 | 每个 compat 一个 notebook |

DICOM-SEG / BIDS 留待后续——接口上不设障碍，新增一个 `DataSource` / `Sink` 实现即可。

---

### 阶段 7 · 文档、发布与收尾（☁️ 文档可做，💻 发布与 PyPI/conda 推送）

| 任务 | 说明 |
|---|---|
| 7.1 用户文档 | 以本文第二部分为骨架，扩充进 `docs/` |
| 7.2 API 参考 | `habit/api/registry.py` 全量符号的 docstring 与自动生成参考 |
| 7.3 迁移指南 | v0.1 → v1.0，含 `migrate-config` 用法与行为差异清单 |
| 7.4 打包发布 | PyPI + conda-forge；`CITATION.cff` 与 Zenodo DOI 联动 |
| 7.5 冻结 v0.1.x | 打 tag，README 标注停止维护 |

---

### 🔴 待补做清单 · L4 配方层与产物落地（阶段 2.5 / 3.2 遗留）

**为什么单列**：阶段 1–6 已合入，L0–L3、执行后端、`compat.*`、报告导出都在。**生境配方链已落地**（`recipes/habitat.py` + `adapters/writers.py` + `cmd_habitat`），但 ML/预处理配方、`run_from_yaml`、checkpoint 透传、可视化、**`habit.core.*` 全删**仍 open；其余 14 个 CLI 命令仍 `import habit.core.*`。

#### A. 配方本体

- [ ] **A1 `Study` 与 `StudyResult` 的生产者**：`StudyResult` 已由 `recipes/habitat.py` / `recipes/result.py` 生产；`Study.fit(cohort, *, backend=None, checkpoint=None)` 对象式入口与 checkpoint 透传仍缺
- [x] **A2 `recipes/habitat.py` 三种范式**：`two_step()` / `one_step()` / `direct_pooling()` — `habit/recipes/habitat.py`（公开名略去 `_habitat` 后缀）
- [x] **A3 `apply_habitat_model()`**：`habit/recipes/habitat.py`
- [x] **A4 `extract_habitat_features()`**：`habit/recipes/features.py`；`cmd_extract_features` 已接线
- [x] **A5 `RunManifest` 汇总**：`recipes/habitat.py::_manifest()` 写 `spec_payload` / `provenance` / seed / `subject_outcomes`
- [ ] **A6 `run_from_yaml()`**：与 CLI 完全等价的 YAML 入口，复用 `LegacyConfigAdapter` / `HabitatSpec`
- [ ] **A7 其余配方**：`traditional_radiomics` ✅ `habit/recipes/features.py` + `cmd_radiomics`；`train_model` / `cross_validate` / `compare_models` / `icc_analysis` / `test_retest_analysis` / `dice` / `preprocess` / `dicom_info` / `sort_dicom` / `merge_tables` 仍缺

#### B. 产物落地（L1 Writer）

- [x] **B1 `adapters/writers.py`**：`DirectoryResultWriter` 实现 `ResultWriter`（`write_habitat_map` / `write_feature_table` / `write_habitat_model`）；写 NRRD 带 `HabitatMap.geometry` spacing/origin/direction
- [x] **B2 `StudyResult.save()` 写生境图**：`habit/recipes/result.py::save()` → `DirectoryResultWriter`；含 `subjXXX_habitats.nrrd`、two-step 的 `subjXXX_supervoxel.nrrd`、`habitats.parquet`
- [ ] **B3 可视化整体未迁移**：v0.1 的聚类 2D/3D 图、交互式 HTML、`cluster_validation.png` 由 `core/habitat_analysis/services/result_publisher.py` + `utils/visualization.py` 产出，新栈无对应物，也未决定它属于 L4 报告导出还是另立协议。**注意绘图不得出现中文**
- [ ] **B4 输出目录结构对齐**：`subjXXX_habitats.nrrd` / `habitats.parquet` ✅；`visualizations/<mode>/...` 仍缺，或明确声明为"允许改善"并写进迁移指南
- [ ] **B5 后处理**：`remove_small_connected_components` 等写图前的后处理在 v0.1 的 image writer 里，需决定归属（L0 kernel 还是 Writer 选项）

#### C. 接线与验收

- [x] **C1 CLI 改调配方**：`habit/commands/cmd_habitat.py` → `habit.recipes.{two_step,one_step,direct_pooling,apply_habitat_model}`
- [x] **C2 golden 比对**：fast synthetic gate ✅ `tests/golden/fast/`（CI：`pytest tests/golden/ -m "not slow"`）；full demo baseline 对 `tests/golden/baseline/` 逐体素仍待验收
- [ ] **C3 文档回填**：第二部分 §9.2 / §9.3 / §10 / §14 的 🔴 标记随实现逐条改回 🟢

#### D. 阶段 1–6 已合入（本清单外原条目）

- [x] **compat.sklearn / monai / nnunet**：`habit/compat/` + `tests/compat/`
- [x] **show_versions**：`habit/api/utils.py`（公开 API `habit.show_versions`）

---

## 4. 测试与 CI 策略

| 层次 | 内容 | 何时跑 |
|---|---|---|
| 架构契约 | 分层依赖、公开符号、打包契约、安装器契约（现有 133 项：`test_architecture_contracts` 41 + `test_public_api` 79 + `test_installer_contracts` 10 + `test_packaging_contracts` 3） | 每次提交 |
| 单元 | `kernels/` 数值核、`contracts/` 不变量、registry 自省 | 每次提交 |
| golden 比对 | 对阶段 0 基线的逐值验证 | 每阶段结束 + 每次合入 `v1.0.0` |
| 配置长尾 | `tests/test_all_configs.py`，62 个流水线配置 × 2 项（schema 校验 + CLI 实跑） | 每日 |
| 端到端 | demo 数据跑通 15 个命令 | 每阶段结束 |

CI 新增门禁：分层依赖违规、公开符号未登记、`import habit` 冷启动超时——三者任一失败即拒绝合入。

---

## 5. 弃用与迁移策略

| 对象 | 策略 |
|---|---|
| 15 个 CLI 命令 | **不弃用**，行为保持 |
| v0 YAML | **不弃用**，`LegacyConfigAdapter` 长期支持 |
| `habit.api.*` 现有 68 个稳定公开符号（`habit.__all__` 为 69，多出的是 `__version__`） | 保留为 v1 新符号的别名，`DeprecationWarning` 提示新位置，v1.x 全程可用 |
| `habitat_pipeline.pkl` | 可读（向后兼容加载），新产物写 `.habitatmodel` |
| 内部私有符号（`_` 前缀、`_PIPELINE_RECIPES` 等） | 直接移除，不做兼容 |

---

## 6. 风险与回滚

| 风险 | 等级 | 缓解 |
|---|---|---|
| 数值漂移 | 高 | 阶段 0 先固化基线；每阶段对齐 |
| 非编程用户体验退步 | 高 | 每次改动跑 §3 阶段 3 的无感验收 |
| 过度抽象 | 中 | 判据见 §1.2 第 3 条 |
| 惰性 `Subject` 与多进程交互 | 中 | 阶段 1 先在 `SerialBackend` 验证序列化边界，阶段 4 再上多进程 |
| 旧 YAML 长尾（74 模板 + 隐式默认） | 中 | `tests/test_all_configs.py` 逐个覆盖 |
| 工程量失控 | 中 | 垂直切片优先；阶段 2 不通过就停 |

每阶段结束打 tag（`v1-phase0` … `v1-phase7`），任何阶段可回退到上一 tag。

---

# 第二部分 · 重构成功后，怎么用 HABIT

> **符号约定**
> ✅ = v0.1.x 已存在、v1.0 保留的能力
> 🆕 = v1.0 新增
> 本部分描述的是 **v1.0 目标状态**。在阶段 7 完成前，🆕 标记的接口以本文为设计契约、以实现为准。
> 第 9 至 12 节的代码块已按当前实现逐条实跑核对，并用 🟢（可直接运行）/ 🔴（尚未实现，照抄会报错）标注，见 §9 开头的说明。

## 7. 四类用户的入口速查

| 你是谁 | 从哪进 | 心智模型 |
|---|---|---|
| 影像组学研究者（医生 / 研究生），只想要结果 | **CLI + YAML**（第 8 节）或 **一行式 API**（第 9.2 节） | "填个配置，跑一条命令" |
| Jupyter 里做探索的研究者 | **场景对象 API**（第 9.3 节） | "拿到对象，一步步看" |
| 方法学开发者，要换掉某个算法做对比 | **领域协议 + registry**（第 9.4 / 11 节） | "在一个个体上把算子调通，实现一个协议，注册进去" |
| 工程集成方 / LLM Agent | **数据契约 + 自省 API**（第 9.5 / 9.6 节） | "内存对象进，内存对象出，schema 可查询" |

四类入口调用的是**同一套核心**，不存在"API 能做但 CLI 不能做"的功能。

---

## 8. CLI 完整参考

### 8.1 安装与自检

```bash
pip install habit                 # 或 conda install -c conda-forge habit
habit --version
habit --help
```

全局选项：`--version`、`-h/--help`。所有子命令均支持 `-h`。

### 8.2 命令总览（15 个现有 ✅ + 1 个新增 🆕）

| 命令 | 用途 | 主要输入 |
|---|---|---|
| `check-config` ✅ | 只校验 YAML 语法与 schema，不跑流程 | `-c` 配置 |
| `sort-dicom` ✅ | 用 dcm2niix 整理 / 重命名 DICOM | `-c` 配置 |
| `dicom-info` ✅ | 提取查看 DICOM 标签信息 | `-i` 目录/文件/配置 |
| `preprocess` ✅ | 批量图像预处理（重采样、配准、归一化…） | `-c` 配置 |
| `get-habitat` ✅ | **生成生境图**（核心命令） | `-c` 配置 |
| `extract` ✅ | 从生境图提取生境特征 | `-c` 配置 |
| `radiomics` ✅ | 传统影像组学特征提取 | `-c` 配置 |
| `model` ✅ | 训练 / 预测机器学习模型 | `-c` 配置 |
| `cv` ✅ | K 折交叉验证 | `-c` 配置 |
| `compare` ✅ | 多模型对比 | `-c` 配置 |
| `icc` ✅ | ICC 一致性分析 | `-c` 配置 |
| `retest` ✅ | 重测信度分析 | `-c` 配置 |
| `dice` ✅ | 两批分割的 Dice 系数 | `--input1/--input2` |
| `merge-csv` ✅ | 按索引列横向合并多个 CSV/Excel | 位置参数 |
| `gui` ✅ | 启动 Web GUI | `--host/--port` |
| `migrate-config` 🆕 | v0 YAML 升级为 v1 格式 | `-c` 配置 |

### 8.3 典型科研全流程

```bash
# 0) 先校验所有配置，避免跑到一半才报错
habit check-config -c config/preprocessing/config_preprocessing_demo.yaml
habit check-config -c config/habitat/config_habitat_two_step.yaml

# 1) DICOM 整理与预处理
habit sort-dicom  -c config/dicom_sort/config_sort_dicom.yaml
habit preprocess  -c config/preprocessing/config_preprocessing_demo.yaml

# 2) 训练集上求生境（产出 HabitatModel）
habit get-habitat -c config/habitat/config_habitat_two_step.yaml -m train

# 3) 用同一个 HabitatModel 映射验证集/外部集
habit get-habitat -c config/habitat/config_habitat_two_step_predict.yaml -m predict

# 4) 生境特征
habit extract -c config/feature_extraction/config_extract_features.yaml

# 5) 建模与评估
habit model   -c config/machine_learning/config_machine_learning.yaml -m train
habit cv      -c config/machine_learning/config_machine_learning_kfold.yaml
habit compare -c config/model_comparison/config_model_comparison.yaml

# 6) 稳定性分析
habit icc    -c config/auxiliary/config_icc_analysis.yaml
habit retest -c config/auxiliary/config_test_retest.yaml
```

### 8.4 逐命令说明

#### `check-config` — 配置校验

```
-c, --config PATH     待校验的 YAML（必填）
-w, --workflow        指定 schema：preprocess|habitat|extract|radiomics|model|
                      cv|compare|icc|retest|sort-dicom（省略时按路径猜测）
    --syntax-only     只校验 YAML 语法（用于 manifest 与 PyRadiomics 参数文件）
```
v1.0 起自动识别 v0 / v1 两套 schema，无需手动切换。

#### `sort-dicom` — DICOM 整理

```
-c, --config PATH     DicomSortConfig YAML（必填）
```
与批量预处理解耦的独立步骤。模板：`config/dicom_sort/config_sort_dicom.yaml`。

#### `dicom-info` — DICOM 信息提取

```
-i, --input PATH              DICOM 目录、单文件或 YAML 配置（必填）
-t, --tags TEXT               逗号分隔的标签，如 "PatientName,StudyDate,Modality"
-o, --output PATH             结果保存路径
-f, --format [csv|excel|json] 输出格式（默认 csv）
    --recursive/--no-recursive        是否递归子目录（默认递归）
    --list-tags                       列出可用标签而非提取
    --num-samples INTEGER             列标签时采样文件数（默认 1）
    --group-by-series/--no-group-by-series   按 SeriesInstanceUID 分组，每序列只读一个文件（默认开）
    --one-file-per-folder             每文件夹只读一个 DICOM，大幅提速
    --dicom-extensions TEXT           识别的扩展名，如 ".dcm,.dicom,.ima"
    --include-no-extension            按魔数识别无扩展名的 DICOM
-j, --num-workers INTEGER             并行线程数，1 表示禁用并行
-d, --max-depth INTEGER               目录递归深度，如 patient/study/series 用 -d 3
```

#### `preprocess` — 图像预处理

```
-c, --config PATH     配置 YAML（必填）
```
可用步骤（详见第 11.1 节）：`load_image`、`dcm2nii`、`resample`、`reorientation`、`registration`、`n4_correction`、`zscore_normalization`、`histogram_standardization`、`adaptive_histogram_equalization`。
模板：`config/preprocessing/` 下 14 个 YAML——10 个可直接跑的流水线配置（含 elastix / WSL / 仅重采样等变体），外加 4 个被它们引用的文件清单（`files_*.yaml`、`image_files.yaml`）。

#### `get-habitat` — 生成生境图（核心）

```
-c, --config PATH             配置 YAML（必填）
-m, --mode [train|predict]    覆盖 YAML 中的运行模式
    --pipeline PATH           predict 模式下覆盖模型路径（v1.0 起接受 .habitatmodel）
    --debug                   调试模式
    --resume                  从个体级检查点续跑 train
```

三种 `clustering_mode`：

| 模式 | 装配方式 | 模板 |
|---|---|---|
| `two_step` | 体素特征 → 超体素化 → 群体级 fit → 逐个体映射 | `config_habitat_two_step*.yaml` |
| `one_step` | 体素特征 →（跳过超体素）→ 个体级 fit + 映射 | `config_habitat_one_step*.yaml` |
| `direct_pooling` | 体素特征 → 直接汇总体素 → 群体级 fit → 逐个体映射 | `config_habitat_direct_pooling*.yaml` |

**train / predict 一致性**：train 产出 `HabitatModel`，predict 加载它把同一套生境定义套到新队列上，绝不重新拟合。这是外部验证成立的前提。

#### `extract` — 生境特征提取

```
-c, --config PATH     配置 YAML（必填）
```
可用特征类型（详见 11.5）：`traditional`、`non_radiomics`、`whole_habitat`、`each_habitat`、`msi`、`ith_score`。

#### `radiomics` — 传统影像组学

```
-c, --config PATH     配置 YAML（必填）
```
PyRadiomics 参数预设：`config/radiomics/parameter*.yaml`、`params_voxel_radiomics.yaml`、`params_supervoxel_radiomics.yaml`。

#### `model` / `cv` / `compare` — 建模、交叉验证、模型对比

```
model:    -c, --config PATH   （必填）
          -m, --mode [train|predict]   覆盖 YAML 的 run_mode（缺省用 YAML，YAML 无该键时为 train）
cv:       -c, --config PATH   （必填）
compare:  -c, --config PATH   （必填）
```
模型路径、数据路径、输出目录全部在配置里指定。可用模型 14 种、特征选择 12 种、指标 9 种，见第 11 节。

#### `icc` / `retest` — 一致性与重测信度

```
-c, --config PATH     配置 YAML（必填）
```
模板：`config/auxiliary/`。

#### `dice` — 分割一致性

```
--input1 PATH         第一批（根目录或配置文件，必填）
--input2 PATH         第二批（必填）
--output TEXT         结果 CSV（默认 dice_results.csv）
--mask-keyword TEXT   掩膜文件夹关键字（默认 masks）
--label-id INTEGER    计算哪个标签（默认 1）
```

#### `merge-csv` — 表格横向合并

```
habit merge-csv FILE1 FILE2 [...] -o merged.csv [选项]

-o, --output PATH     输出路径（必填）
-c, --index-col TEXT  索引列名。单个名字对所有文件生效；也可逗号分隔逐文件指定
                      例："id" 或 "PatientID,subject_id"
    --separator TEXT  分隔符（默认 ,）
    --encoding TEXT   编码（默认 utf-8）
    --join [inner|outer]  连接方式（默认 inner）
```

#### `gui` — Web 界面

```
    --host TEXT       绑定地址（默认 127.0.0.1）
-p, --port INTEGER    端口（默认 8501）
    --no-browser      不自动开浏览器
```

#### `migrate-config` 🆕 — 配置升级

```
-c, --config PATH     待升级的 v0 YAML（必填）
-o, --output PATH     输出的 v1 YAML（默认原地加 .v1 后缀）
    --dry-run         只打印差异不写文件
```
不升级也能继续跑，本命令只是让你享受 v1 的新字段与更清晰的分区。

---

## 9. Python API 完整参考

> **本章代码块的实现状态标记**（截至阶段 6 完成，对照 `py310` 环境实测）
> 🟢 = 当前实现可直接运行，示例已核对签名
> 🔴 = 目标设计，**当前实现中尚不存在**，照抄会报错
> 未标记的段落为说明性文字。
> 阶段 1–6 已合入（L0–L3、执行后端、compat、报告导出），**L4 配方层 `habit.recipes` 尚未落地**，因此 §9.2 与 §9.3 中依赖配方的部分标 🔴。

### 9.1 分层与稳定性承诺

```
habit.recipes     L4     一行式研究配方          ← 研究者最常用（🔴 尚未落地）
habit.<顶层便捷>   L4     配方对象与结果对象       ← notebook 探索
habit.domain      L3     八个领域协议 + 内置实现  ← 方法学开发者
habit.contracts   L2     领域数据模型             ← 工程集成
habit.adapters    L1     DataSource / Sink
habit.kernels     L0     纯计算（部分稳定公开）    ← 想直接复核公式的人
habit.registry           注册表基类与 entry point ← 插件作者
habit.<自省函数>          list_plugins 等四函数    ← GUI / LLM Agent
habit.execution          执行后端
habit.spec               Spec / RunPolicy / YAML 同构
habit.compat.*           sklearn / MONAI / nnU-Net 适配
```

稳定性：`import habit` 直接暴露的符号（登记在 `habit/api/registry.py`）遵循语义化版本，v1.x 内不破坏；下划线开头的一切视为内部实现。

`habit.kernels` **选择性稳定**：有明确科学定义、会被论文引用和复核的公式（MSI、ITH score 等）作为稳定公开 API 承诺不破坏；其余数值内核（各类中间实现、缓存结构）标为内部，可能随重构变化。理由是方法学开发者常常想在裸数组上直接验证 HABIT 的指标算得对不对，把这条路堵死会削弱可信度：

```python
from habit.kernels.habitat_metrics import (
    ith_score,
    spatial_interaction_matrix,
    msi_features_from_matrix,
)

score = ith_score(habitat_label_array)                              # -> float
matrix = spatial_interaction_matrix(habitat_label_array, n_classes=4)
features = msi_features_from_matrix(matrix)                         # -> Dict[str, float]
```

MSI 拆成两步（先算空间交互矩阵，再由矩阵导出特征），是为了让矩阵本身也能被单独复核。`habit.kernels.habitat_metrics` 当前导出的稳定符号为 `ith_score`、`spatial_interaction_matrix`、`msi_features_from_matrix`、`habitat_region_stats`、`habitat_volume_fractions`。这些函数只吃 numpy 数组，不涉及 `Subject`、配置或文件系统。

### 9.2 一行式：给只想要结果的研究者

🔴 `habit.recipes` 是 L4 配方层，阶段 2.5 的 `recipes/habitat.py` 尚未落地，本节全部代码目前**跑不通**。在配方层补齐之前，等价能力请走 §9.4 手工装配，或继续用 v0.1 的 `habit.run_habitat_analysis(...)`。

```python
import habit

# Read a cohort from a conventional directory layout and run the full
# two-step habitat workflow with all defaults.
# A recipe builds the STUDY (what to do); fit() runs it on a COHORT (on which data).
# The separation is what makes the same study reusable on an external cohort.
study = habit.recipes.two_step_habitat(
    modalities=["T1", "T1C", "T2"],
    n_supervoxels=50,
    n_habitats=4,
    habitat_features=["msi", "ith_score"],
    random_seed=42,
)
cohort = habit.Cohort.from_directory(
    "D:/study/preprocessed", modalities=["T1", "T1C", "T2"], roi="tumor",
)
result = study.fit(cohort)                    # -> StudyResult

print(result.habitat_model.summary())         # human-readable model card
result.features.frame.to_csv("feat.csv")      # FeatureTable wraps a pandas frame
print(result.manifest.describe_methods())     # auto-generated methods paragraph
result.save("D:/study/out")                   # writing to disk is a separate, explicit act
```

其中 `Cohort.from_directory`、`StudyResult`（字段 `habitat_model` / `pipeline` / `features` / `habitat_maps` / `manifest`）、`HabitatModel.summary()`、`RunManifest.describe_methods()`、`StudyResult.save(out_dir)` 都已实现 🟢，缺的只是产出 `StudyResult` 的那个配方函数。注意模型卡的方法名是 **`summary()`**（statsmodels 惯例）而不是 `describe()`——后者在科学 Python 里已被 `DataFrame.describe()` 占用为统计表。

同族配方还有 `one_step_habitat`、`direct_pooling_habitat`、`apply_habitat_model`、`extract_habitat_features`、`traditional_radiomics`、`train_model`、`cross_validate`、`compare_models`。**每个 CLI 命令都有一个对应配方**，参数与 YAML 字段一一对应。

只有一种调用形态：**配方造 study，`study.fit(cohort)` 出 result**。不提供"传目录直接出结果"的一步式重载——那会把"做什么分析"和"在哪份数据上做"重新粘回去，正是 v0.1 让外部验证难做的原因。

从 YAML 直接驱动（与 CLI 完全等价）：

```python
result = habit.recipes.run_from_yaml("config/habitat/config_habitat_two_step.yaml")
```

### 9.3 场景对象：给 notebook 里做探索的人

🟢 队列侧全部可用：

```python
from pathlib import Path
from habit import Cohort, HabitatModel

cohort = Cohort.from_directory(
    "D:/study/preprocessed",
    modalities=["T1", "T1C", "T2"],
    roi="tumor",
    images_folder="images",   # 默认值，目录约定不同时再改
    masks_folder="masks",
)
print(len(cohort), cohort.subject_ids[:5])

train = cohort.filter(lambda s: s.metadata["center"] == "A")
test = cohort.filter(lambda s: s.metadata["center"] == "B")

model = HabitatModel.load(Path("habitat_k4.habitatmodel"))
pipeline = ...                           # 见 §9.4.4，由手工装配得到

maps_train = train.map(pipeline)         # Sequence[HabitatMap], cohort order
maps_test = test.map(pipeline)           # exactly the same habitat definition
one_map = pipeline(test[0])              # drop down to a single subject any time

model.save("habitat_k4.habitatmodel")
```

🔴 用配方一步得到 `pipeline` 的写法（等配方层落地后才可用）：

```python
study = habit.recipes.two_step_habitat(
    modalities=["T1", "T1C", "T2"], n_supervoxels=50, n_habitats=4, random_seed=42,
)
result = study.fit(train)     # .habitat_model (definition) + .pipeline (procedure)
maps_test = test.map(result.pipeline)
```

`Cohort` 是**真正的数据对象**（可迭代 / 切片 / filter / 惰性持有影像），不是对目录字符串的包装。

### 9.4 领域协议：给要换算法做对比的方法学开发者

#### 9.4.1 单个体是原子调用

**这是 API 的最底层，也是调试与方法比较的起点。** 每个个体级算子都是单参可调用对象，处理一个个体不需要队列、不需要执行后端、不需要配置文件：

🟢

```python
from habit.domain import VoxelFeatureExtractorRegistry, SupervoxelizerRegistry

voxel_fx = VoxelFeatureExtractorRegistry.create("raw", modalities=["T1", "T1C", "T2"])
svx      = SupervoxelizerRegistry.create("slic", n_supervoxels=50)

field = voxel_fx(subject)     # Subject           -> VoxelFeatureField
unit  = svx(field)            # VoxelFeatureField -> Supervoxelization
```

等价的直接构造（IDE 补全更友好，无魔法字符串）：

```python
from habit.domain import RawVoxelFeatures, SlicSupervoxelizer

voxel_fx = RawVoxelFeatures(modalities=["T1", "T1C", "T2"])
svx      = SlicSupervoxelizer(n_supervoxels=50)
```

构造方式沿用 v0.1.x 已有的注册表写法 `<Registry>.create(name, **params)`——注册表类本身就代表了组件族，因此不需要再传一个 `kind` 字符串，也就没有第二个魔法字符串可以拼错。字符串驱动的路径留给 YAML 与 Agent，见 §9.6。注册表**统一以 `Registry` 结尾**，没有 `Factory` 后缀的类。各注册表当前可用的名字：

| 注册表 | `available()` |
|---|---|
| `VoxelFeatureExtractorRegistry` | `raw` |
| `FeaturePreprocessingMethodRegistry` | `impute`、`minmax`、`zscore`、`robust`、`log`、`winsorize`、`binning`、`variance_filter`、`correlation_filter` |
| `SupervoxelizerRegistry` | `slic`、`kmeans`、`gmm` |
| `SupervoxelFeatureExtractorRegistry` | `mean_voxel_features`、`supervoxel_radiomics` |
| `HabitatModelFitterRegistry` | `kmeans`、`gmm` |
| `HabitatAssignerRegistry` | `nearest_centroid` |
| `HabitatFeatureExtractorRegistry` | `msi`、`ith_score`、`volume`、`non_radiomics`、`traditional`、`whole_habitat`、`each_habitat` |

`FeaturePreprocessingMethodRegistry` 注册的是**方法**而不是协议实现：两个预处理协议的实现都是链（方法的有序组合），可插拔的粒度落在方法上，所以域名叫 `feature_preprocessing_method`。用 `build_methods([...])` 从 spec 列表批量构造。

🔴 领域动词别名（`voxel_fx.extract(subject)` 等价于 `voxel_fx(subject)`）目前**未实现**，且已否决——类体里的 `extract = __call__` 绑定定义时刻的函数对象，子类覆写 `__call__` 后别名仍指向父类实现，是个只会算错不会报错的陷阱。算子只暴露 `__call__`。

调用约定本身与 MONAI 的 `transform(data)`、TorchIO 的 `transform(subject)`、PyRadiomics 的 `extractor.execute(img, mask)` 同源。

#### 9.4.2 八个协议

🟢 `habit.domain.protocols` 的实际签名，按流水线顺序：

```python
VoxelFeatureExtractor        __call__(Subject)                          -> VoxelFeatureField
SubjectFeaturePreprocessor   __call__(DataFrame)                        -> DataFrame
Supervoxelizer               __call__(VoxelFeatureField)                -> Supervoxelization
SupervoxelFeatureExtractor   __call__(Subject, Supervoxelization)       -> Supervoxelization
CohortFeaturePreprocessor    fit(DataFrame) / transform(DataFrame)      -> DataFrame           ← 群体级
HabitatModelFitter           fit(Sequence[Supervoxelization], *, cohort=None) -> HabitatModel  ← 群体级
HabitatAssigner              __call__(Supervoxelization)                -> HabitatMap          ← 个体级
HabitatFeatureExtractor      __call__(Subject, HabitatMap)              -> FeatureTable
```

命名两点要注意：群体级建模那个叫 **`HabitatModelFitter`** 而不是 `*Estimator`（sklearn 把 estimator 保留给 `fit` 返回 `self` 且可 `clone()` 的对象，而这里的 `fit` 返回一个全新的 `HabitatModel` 制品，是 lifelines / statsmodels 的 `*Fitter` 语义）；个体级那个叫 **`HabitatAssigner`** 而不是 `HabitatMapper`。

`SupervoxelFeatureExtractor` 是第六个协议：它替换超体素的默认描述（特征均值），per-supervoxel radiomics 需要原始强度，而 `VoxelFeatureField` 故意不携带强度，所以只能单开一个协议。

两个预处理协议是第七、第八个。它们的分界不是数据粒度而是**状态是否跨受试者**：`SubjectFeaturePreprocessor` 无状态、逐个体、消除个体间差异；`CohortFeaturePreprocessor` 有状态、在训练队列上 fit、让不同个体落进同一个可比空间。因此**同一个无状态链可以同时用在体素特征和超体素特征上**——这是 v0.1 表达不出来的（它的个体级块只作用于体素）。

它们收 `DataFrame` 而不是契约类型，因为按列的数值运算真的与行的含义无关。契约到 `DataFrame` 的桥是两个契约各自提供的对称方法对：

```python
frame = field.feature_frame()                       # 裸特征矩阵，无坐标列
field = field.with_feature_frame(                   # 装回去并派生溯源
    frame, produced_by="...", spec_fingerprint="..."
)
```

`Supervoxelization` 有同名的一对。所以 `VoxelFeatureField`（`ndarray` + 列名）与 `Supervoxelization`（带 index 的 `DataFrame`）存储方式不同，但对算法呈现同一个接口。

群体级有两个协议：`CohortFeaturePreprocessor` 与 `HabitatModelFitter`。这是生境分析的本质（生境定义必须跨个体可比），不是设计限制。

`HabitatAssigner` 在**构造期**接收模型，因此它也是普通单参调用；同时"没 fit 就 predict"变成构造不出来的状态，而不是运行期报错。常用写法：

```python
assigner = model.assigner()                    # 工厂方法，默认 "nearest_centroid"
assigner = model.assigner("nearest_centroid")  # 或显式指定实现与参数
habitat_map = assigner(unit)
```

#### 9.4.3 手工装配整条链

🟢 等价于内置的 `two_step`，但每一步都可替换：

```python
from habit.domain import HabitatModelFitterRegistry

fitter = HabitatModelFitterRegistry.create("kmeans", n_habitats=4)
fitter.set_random_state(42)

fields = [voxel_fx(s) for s in cohort]        # or cohort.map(voxel_fx)
units  = [svx(f) for f in fields]
model  = fitter.fit(units, cohort=cohort)     # the only cohort-level step
maps   = [model.assigner()(u) for u in units]
```

`fit` 的 `cohort` 是可选的，只用于在模型里记录一份不可识别的队列指纹（`CohortFingerprint`）；省略它模型照样能 fit，但产出的制品就少了来源描述。`n_habitats` 留空时由 `min_habitats` / `max_habitats` / `validation`（默认 `silhouette`）自动选簇数。

想加第四种范式？换个装配顺序即可，**不需要改 HABIT 源码**——这是与 v0.1 硬编码 `_PIPELINE_RECIPES` 的根本区别。

带上特征预处理的完整版本如下。个体级链每次从当前矩阵重算统计量，队列级链只在**训练**单元上 fit 一次：

🟢

```python
import pandas as pd
from habit.domain import (
    CohortPreprocessingChain, SubjectPreprocessingChain, build_methods,
)

# 个体级：无状态，用于消除个体间差异（这里作用于体素特征）
voxel_chain = SubjectPreprocessingChain(build_methods([
    {"name": "winsorize", "params": {"winsor_limits": [0.05, 0.05]}},
    {"name": "minmax"},
]))
# 队列级：有状态，让不同个体的超体素落进同一个可比空间
cohort_chain = CohortPreprocessingChain(build_methods([{"name": "zscore"}]))

units = []
for subject in cohort:
    field = voxel_fx(subject)
    field = field.with_feature_frame(
        voxel_chain(field.feature_frame()),
        produced_by="feature_preprocessing.subject.voxel",
        spec_fingerprint=voxel_chain.spec.fingerprint(),
    )
    units.append(svx(field))

pooled = pd.concat([u.feature_frame() for u in units], ignore_index=True)
cohort_chain.fit(pooled)                      # 只看训练数据
units = [
    u.with_feature_frame(
        cohort_chain.transform(u.feature_frame()),
        produced_by="feature_preprocessing.cohort",
        spec_fingerprint=cohort_chain.spec.fingerprint(),
    )
    for u in units
]

model = fitter.fit(units, cohort=cohort)
model = model.with_cohort_preprocessing(       # 特征空间随模型走
    cohort_chain.state, cohort_chain.spec.to_dict()
)
```

最后那一步不是可选的：质心只在预处理后的特征空间里有意义，不把链绑进模型，别人拿去套新队列时会用原始特征比对预处理后的质心，而且照样会返回一张看起来合理的生境图。`with_cohort_preprocessing()` 因此也会重算 `model_id`。

两条链都会在未显式配置 `impute` 时**自动前置一个默认实例**（`strategy="mean"`），因为其余方法都假定输入有限。自动加的这一步会出现在 `chain.spec` 里——溯源不能有"记录之外的计算"。想改策略或改位置就显式写出来：

```python
SubjectPreprocessingChain(build_methods([
    {"name": "impute", "params": {"strategy": "median"}},
    {"name": "robust"},
]))
```

#### 9.4.4 `SubjectPipeline`：把个体级链条合成一个可调用对象

HABIT 版的 `Compose`。因为各步类型异构（`Subject → VoxelFeatureField → Supervoxelization → HabitatMap`），用的是保留类型的具名管线而不是泛型 Compose：

🟢

```python
from habit import SubjectPipeline
from habit.domain import HabitatFeatureExtractorRegistry

pipeline = SubjectPipeline(
    voxel_feature_extractor=voxel_fx,
    supervoxelizer=svx,                     # None 表示直接聚类体素（one_step / direct_pooling）
    habitat_assigner=model.assigner(),
    supervoxel_feature_extractor=None,          # 可选，替换超体素的默认特征均值
    voxel_feature_preprocessor=voxel_chain,     # 可选，无状态，在超体素化之前
    supervoxel_feature_preprocessor=None,       # 可选，无状态，在超体素特征之后
    cohort_feature_preprocessor=cohort_chain,   # 已 fit 的队列级链，在分配之前
)

msi = HabitatFeatureExtractorRegistry.create("msi")        # 内置生境特征族均无构造参数
ith = HabitatFeatureExtractorRegistry.create("ith_score")

habitat_map = pipeline(subject)                             # one subject, end to end
table = pipeline.extract_features(subject, [msi, ith])      # plus habitat features
maps = cohort.map(pipeline)                                 # whole cohort
```

参数名是 **`habitat_assigner`**；`supervoxel_feature_extractor` 与 `supervoxel_feature_preprocessor` 只在 `supervoxelizer` 非 `None` 时有意义（单个体素没有区域可描述），两者矛盾时构造期直接报 `HABITAPIError`。

`voxel_feature_preprocessor` 的位置很关键：它在**超体素化之前**。先按个体归一化再划分区域，超体素边界才不会去追扫描仪的强度尺度。

**训练期用同一个 `SubjectPipeline`，只是不给 assigner。** `habitat_assigner=None` 是合法状态，表示这条链只产出聚类单元：

🟢

```python
fit_pipeline = SubjectPipeline(
    voxel_feature_extractor=voxel_fx,
    supervoxelizer=svx,
    habitat_assigner=None,                      # 训练期还没有模型
    voxel_feature_preprocessor=voxel_chain,
)
units = [fit_pipeline.units(s) for s in cohort]  # 喂给 fitter
# pipeline(subject) 此时会报错，提示先 fit 再用 model.assigner() 重建管线
```

`pipeline.units(subject)` 与 `pipeline(subject)` 共用同一段实现，前者是后者去掉最后的分配步骤。训练和预测各写一遍装配，正是两者悄悄分岔的标准途径——共用一个对象就是为了让这件事不可能发生。

外部验证要分发的东西正好是 `HabitatModel` + `SubjectPipeline`：生境定义，加上套用这个定义的过程。

#### 9.4.5 随机性控制

生境分析对种子异常敏感（k-means / GMM 初始化，群体级聚类还对受试者顺序敏感）。随机组件统一实现 `set_random_state(seed)`，不再各自发明 `random_state` 参数：

```python
KMeansSupervoxelizer(...).set_random_state(42)      # 随机初始化 -> Seedable
GmmSupervoxelizer(...).set_random_state(42)
KMeansHabitatModelFitter(...).set_random_state(42)
```

**确定性组件不实现该协议**——这本身就是会被写进溯源记录的信息。具体地，`SlicSupervoxelizer` 是确定性的，**没有** `set_random_state`，对它调用会抛 `AttributeError`；判定方式是 `isinstance(component, Seedable)`。

### 9.5 数据契约：给工程集成方

| 类型 | 职责 | 关键不变量 |
|---|---|---|
| `Geometry` | spacing / origin / direction / shape | 参与运算的空间对象必须共享同一 geometry |
| `ImageVolume` / `MaskVolume` | 带几何的影像与掩膜 | 不可变，与 geometry 强绑定 |
| `ImageRef` | 影像的惰性引用 | `.load()` 时才读盘；可自行实现（PACS / zarr / torch） |
| `Subject` | 一个受试者：`{模态: 影像}` + `{ROI: 掩膜}` + 元数据 | 影像字段惰性 |
| `Cohort` | `Subject` 的有序容器 | `subject_id` 唯一 |
| `VoxelFeatureField` | ROI 内每体素一个特征向量 | 行数 = ROI 体素数 |
| `Supervoxelization` | 个体内超体素划分 + 每超体素特征 | 标签覆盖整个 ROI，无空洞 |
| `HabitatMap` | 生境标签图 | 记录来源 `model_id` |
| **`HabitatModel`** | **群体级生境定义** | 自足：脱离训练输出目录即可套用新队列 |
| `FeatureTable` | 特征表 + 列语义 | 明确区分 ID 列 / 特征列 / 标签列 |
| `Provenance` | 该对象的来源 | 沿数据流传播，不丢失 |

完全内存驱动（不碰文件系统）：

```python
import numpy as np
from habit.contracts import Subject, Cohort, ArrayImageRef, Geometry

# shape 是 NumPy 轴序 (z, y, x)；spacing / origin / direction 是 SimpleITK 轴序 (x, y, z)
geom = Geometry(shape=(64, 128, 128), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
                direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
t1_array = np.random.rand(64, 128, 128)
t2_array = np.random.rand(64, 128, 128)
mask_array = (np.random.rand(64, 128, 128) > 0.7).astype(np.int32)  # 掩膜必须是整数标签，0 为背景

subject = Subject(
    subject_id="P001",
    images={"T1": ArrayImageRef(t1_array, geom), "T2": ArrayImageRef(t2_array, geom)},
    masks={"tumor": ArrayImageRef(mask_array, geom)},
    metadata={"center": "A"},
)
cohort = Cohort([subject])
```

`Subject.images` / `Subject.masks` 存的是 `ImageRef`（惰性引用），`ArrayImageRef` 是内存数组的那个实现，`subject.image("T1")` / `subject.mask("tumor")` 时才 materialise 成 `ImageVolume` / `MaskVolume`。

如果要直接构造已 materialise 的卷，注意 `ImageVolume` / `MaskVolume` 继承自稳定公开类 `habit.api.image`，构造器把几何**展开成 `spacing` / `origin` / `direction`** 而不是收成一个 `Geometry`。手上已经有 `Geometry` 对象时用 `from_geometry`：

```python
from habit.contracts import ImageVolume, MaskVolume

image = ImageVolume.from_geometry(t1_array, geom, modality="T1")
mask = MaskVolume.from_geometry(mask_array, geom, roi_name="tumor")

image.geometry            # -> Geometry，与 geom 相等
image.load()              # -> np.ndarray，ImageRef 协议的另一半
```

`HabitatModel` 的内容与持久化：

```
model_id            稳定标识（含 spec 指纹）
n_habitats          生境数
feature_names       群体级聚类所用特征的名称与顺序
centroids           群体质心
preprocessing_state 训练期学到的状态（binning 边界、归一化参数…）
spec_payload        产生它的完整算法规格（可导出 YAML）
cohort_fingerprint  来源队列描述：n、模态、来源、伦理与可分享性声明
provenance          软件版本、依赖版本、随机种子、时间戳
```
`save()` / `load()` 使用**版本化自描述格式**，跨 HABIT 版本要么可读，要么给出明确的不兼容提示——不是裸 pickle。

### 9.6 自省 API：给 GUI 与 LLM Agent

**沿用 v0.1.x 已有的四个函数，不另造同义 API。** 它们在 v0.1.x 就是公开符号，v1.0 只是把覆盖面补全并增加 JSON Schema 导出：

🟢

```python
from habit import list_plugins, get_plugin_info, get_param_schema, load_plugins

load_plugins()                                  # discover third-party entry points

list_plugins()                                  # -> Tuple[PluginInfo, ...]，全部域
list_plugins("habitat_model_fitter")
# (PluginInfo(name='gmm', ...), PluginInfo(name='kmeans', ...))

info = get_plugin_info("kmeans", "habitat_model_fitter")
# PluginInfo(name='kmeans', domain=..., implementation=..., params_schema=..., provider='built-in')

get_param_schema("kmeans", "habitat_model_fitter")     # Pydantic model class or None
get_param_schema("kmeans", "habitat_model_fitter").model_json_schema()  # JSON Schema
```

注意参数顺序是 `(name, domain)`，与 v0.1.x 一致。**域名不是一律复数**：v1.0 新增的领域协议域用单数，v0.1.x 遗留的域保留原来的复数名。当前全部可用域为

- v1.0 领域协议（单数）：`voxel_feature_extractor`、`supervoxelizer`、`supervoxel_feature_extractor`、`habitat_model_fitter`、`habitat_assigner`、`habitat_feature_extractor`
- v1.0 聚类输入预处理：`feature_preprocessing_method`（唯一一个域名不等于协议名的域，理由见 `08` §4：两个预处理协议的实现是链，可插拔的是链里的方法）
- v1.0 表格 ML（单数）：`table_preprocessor`、`feature_selector`、`classifier`、`metric`

`feature_preprocessing_method` 与 `table_preprocessor` 数值实现相同但**不是一个域**：前者的一行是一个体素或超体素（通往生境定义），后者的一行是一个受试者（通往结局模型）。
- v0.1.x 遗留（复数）：`feature_extractors`、`habitat_features`、`metrics`、`models`、`preprocessors`

传错域名不会静默返回空，而是抛 `HABITAPIError` 并列出全部合法域。v1.0 的两处扩展：

1. 补齐三个尚无插件域的注册表（聚类算法、特征表预处理、特征选择），并为新增的领域协议加上对应域；
2. `get_param_schema` 返回 Pydantic 模型类（无参数组件返回空 schema 的模型，未注册时返回 `None`），配合 `.model_json_schema()` 直通 JSON Schema，供 GUI 表单与 Agent 校验使用。

LLM Agent 可据此**自动构造合法的 spec**，不需要预先了解 HABIT 源码。GUI 的表单由同一份 schema 生成，保证 CLI / API / GUI 三者能力一致。

### 9.7 执行后端

🟢 **后端是可选加速器，不是前置条件。** 三层递进，任何一层都能独立干完活：

```python
field  = voxel_fx(subject)                     # 1) one subject, no infrastructure
fields = cohort.map(voxel_fx)                  # 2) whole cohort, serial by default
fields = cohort.map(voxel_fx, backend=...)     # 3) opt in to parallelism
```

不构造后端也能跑通全流程。只有在真的需要并行、单例超时或断点续跑时才走第三层：

```python
from habit.execution import SerialBackend, ProcessPoolBackend, CheckpointStore

backend = ProcessPoolBackend(
    workers=8,
    subject_timeout_sec=1800.0,
    on_subject_failure="continue",    # or "fail_fast"
    oom_backoff=True,
    oom_reduce_workers_by=2,
    cap_workers_to_gpu_pool=True,
)
checkpoint = CheckpointStore("D:/study/ckpt")

maps = cohort.map(pipeline, backend=backend, checkpoint=checkpoint)
```

`ProcessPoolBackend` 的其余参数（均有默认值）：`subject_spawn_timeout_sec`、`graceful_shutdown_sec`、`parallel_mode`、`auto_retry_rounds`、`resume`、`retry_failed_subjects`、`force_rerun_subjects`、`clear_checkpoint_on_success`。`SerialBackend` 只有一个 `on_subject_failure`。

算子只声明"我是个体级、可并行的"，**不自己管进程池**。将来接 Dask / Ray / 集群调度只是多一个 `ExecutionBackend` 实现，算法代码零改动。

🔴 **配方层怎么传？** 配方内部也走同一条路，只是把 `backend` / `checkpoint` 往下透传（依赖尚未落地的配方层）：

```python
result = study.fit(cohort, backend=backend, checkpoint=checkpoint)
```

CLI 与 YAML 走的是 `RunPolicy`——它是这些执行参数的**声明式快照**，由适配层翻译成后端对象。所以只有两种写法：Python 里给对象，YAML 里给 `RunPolicy` 字段，两者字段名一一对应（`workers`、`subject_timeout_sec`、`on_subject_failure`、`oom_backoff`、`resume`、`checkpoint_dir`）。

### 9.8 溯源与报告

🟢 `RunManifest` 本身已实现（`result.manifest` 这条获取途径依赖配方层）：

> ⚠ **重名陷阱**：本节说的是 `habit.contracts.RunManifest`。顶层 `habit.RunManifest` 是 v0.1 的 `habit.api.provenance.RunManifest`，只有 `git_commit` / `to_dict`，**没有** `describe_methods` / `checklist`。同理 `habit.create_run_manifest()` 产出的是 v0.1 那个。要用本节能力请显式 `from habit.contracts import RunManifest`。

```python
from habit.contracts import RunManifest

manifest = result.manifest                      # 或自行构造 RunManifest(spec_payload=..., provenance=...)

manifest.software_versions()                    # habit / numpy / SimpleITK / PyRadiomics ...
manifest.random_seeds()
manifest.describe_methods(style="radiology")    # ready-to-paste English methods paragraph；另一种是 "nature"
manifest.checklist("IBSI")                      # item-by-item compliance table -> pd.DataFrame
manifest.checklist("CLEAR")
manifest.checklist("METRICS")
manifest.checklist("TRIPOD+AI")
manifest.to_json("run_manifest.json")           # 省略路径则只返回 JSON 字符串
```

`checklist` 支持的标准恰好是 `IBSI` / `CLEAR` / `METRICS` / `TRIPOD+AI`，`describe_methods` 支持的风格是 `radiology` / `nature`，传其它值抛 `HABITAPIError`。

`Provenance` 随数据流自动传播，因此方法学描述不是事后手写的，而是从实际执行的算子链**推导**出来的。这是 v1.0 相对同类工具的主要差异点，也是许可证转为宽松后保障学术可追溯性的工程手段。

### 9.9 异常体系

统一继承自 `HABITAPIError`，便于集成方精确捕获：

`HabitError`、`ConfigurationError`、`DataFormatError`、`GeometryError`、`OptionalDependencyError`、`ComponentNotFoundError`、`CompatibilityError`、`ProcessingError`、`NotFittedError`

🟢 十个异常类型全部可从 `habit` 顶层导入：

```python
try:
    habitat_map = pipeline(subject)
except habit.GeometryError as exc:
    ...   # image / mask geometry mismatch
except habit.OptionalDependencyError as exc:
    ...   # e.g. elastix or AutoGluon not installed
```

### 9.10 生态适配 `habit.compat.*`

全部为**可选依赖**，不装也不影响核心。

**sklearn**：

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from habit.compat.sklearn import as_estimator

pipe = Pipeline([
    ("habitat", as_estimator(habitat_spec)),
    ("clf", LogisticRegression()),
])
GridSearchCV(pipe, {"habitat__n_habitats": [3, 4, 5]}, cv=5).fit(cohort, y)
```

`habit.compat.sklearn` 实际导出：`as_estimator(spec, **overrides)`、`as_transformer(component)`、`as_classifier(component)`，及其对应的三个类 `HabitatFeaturesEstimator` / `TableTransformerEstimator` / `TableClassifierEstimator`。

**sklearn 的命名分歧（有意为之）**：`HabitatModelFitter.fit()` 返回 `HabitatModel` 而不是 `self`，违反 sklearn 惯例。这也正是它叫 `*Fitter` 而不是 `*Estimator` 的原因——生境定义本身就是科学产物，不是估计器的内部状态，它要能被 `save()`、被分发、被论文引用。`compat.sklearn` 负责桥接这个分歧，领域 API 不迁就。

**MONAI / PyTorch**：因为个体级算子就是单样本可调用对象，HABIT 算子可以**直接**当 MONAI transform 用，用户继续用自己的 `DataLoader` 做并行，不必把执行控制权交给 HABIT：

```python
from monai.data import Dataset, DataLoader
from monai.transforms import Compose
from habit import SubjectPipeline
from habit.compat.monai import AsMonaiDict, to_monai_dict, from_monai_dict

to_habitat_map = SubjectPipeline(
    voxel_feature_extractor=voxel_fx,
    supervoxelizer=svx,
    habitat_assigner=model.assigner(),
)

transform = Compose([AsMonaiDict(), to_habitat_map])     # a HABIT op as a MONAI transform
loader = DataLoader(Dataset(list(cohort), transform=transform), num_workers=4)
```

**nnU-Net**：

```python
from habit.compat.nnunet import NnUNetDataSource

# 方法名是 load()（DataSource 协议），不是 cohort()
cohort = NnUNetDataSource("Dataset001_Tumor", roi_label=1).load()  # imagesTr/labelsTr + dataset.json
```
即用 nnU-Net 的分割结果直接作为 HABIT 的输入掩膜，无需手工搬文件。

---

## 10. CLI 与 API 的对应关系

每个命令都能在 API 里找到等价物，三条路径共享同一核心。🔴 "配方函数"整列依赖尚未落地的 `habit.recipes`；"底层协议/组件"列已实现：

| CLI | 配方函数 | 底层协议/组件 |
|---|---|---|
| `preprocess` | `recipes.preprocess()` | 图像预处理器组件链（`preprocessors` 域） |
| `get-habitat` | `recipes.two_step_habitat()` / `one_step_habitat()` / `direct_pooling_habitat()`，predict 模式对应 `apply_habitat_model()` | 八个领域协议 |
| `extract` | `recipes.extract_habitat_features()` | `HabitatFeatureExtractorRegistry` |
| `radiomics` | `recipes.traditional_radiomics()` | PyRadiomics 内核 |
| `model` | `recipes.train_model()` / `predict()` | `ClassifierRegistry` + `FeatureSelectorRegistry` + `TablePipeline` |
| `cv` | `recipes.cross_validate()` | 同上 |
| `compare` | `recipes.compare_models()` | `MetricRegistry` + DeLong |
| `icc` | `recipes.icc_analysis()` | ICC 内核 |
| `retest` | `recipes.test_retest_analysis()` | 同上 |
| `dice` | `recipes.dice()` | 分割一致性内核 |
| `dicom-info` | `recipes.dicom_info()` | pydicom 适配 |
| `sort-dicom` | `recipes.sort_dicom()` | dcm2niix 适配 |
| `merge-csv` | `recipes.merge_tables()` | `FeatureTable` 合并 |
| `check-config` | `spec.validate_yaml()` | schema 校验 |
| `migrate-config` | `spec.migrate_yaml()` | `LegacyConfigAdapter` |

---

## 11. 全部可插拔组件清单

共 70+ 个内置实现，通过 `list_plugins(domain)` 查询。表头给出的是**插件域名**，也就是 entry point 分组 `habit.<域名>` 的后半段。

标注 ⚠ 的注册表在 v0.1.x **尚无插件域**，外部包装不进来，v1.0 补齐（见 §12.2）。标注「模板」的条目是供复制的骨架，多数默认不加载，需要手动 import 才会注册。

🟢 **域名以实现为准**（`list_plugins()` 实测）：v1.0 新增域用单数，v0.1.x 遗留域保留复数，两者并存且部分内容重叠：

| v1.0 域名（单数） | v0.1.x 遗留别名（复数） | 本节 |
|---|---|---|
| `preprocessor` | `preprocessors` | §11.1 |
| `voxel_feature_extractor` / `supervoxel_feature_extractor` | `feature_extractors` | §11.2 |
| `supervoxelizer` / `habitat_model_fitter` / `habitat_assigner` | 无 | §11.3 |
| `feature_preprocessing_method` | 无 | §11.3b |
| `table_preprocessor` | 无 | §11.4 |
| `habitat_feature_extractor` | `habitat_features` | §11.5 |
| `classifier` | `models` | §11.6 |
| `feature_selector` | 无 | §11.7 |
| `metric` | `metrics` | §11.8 |

### 11.1 图像预处理 `domain="preprocessor"`（遗留别名 `"preprocessors"`）

| 名称 | 说明 |
|---|---|
| `load_image` | 载入影像（管线起点） |
| `dcm2nii` | dcm2niix 转换 |
| `resample` | 重采样到目标 spacing |
| `reorientation` | 方向标准化 |
| `registration` | 配准（SimpleITK / elastix） |
| `n4_correction` | N4 偏置场校正 |
| `zscore_normalization` | Z-score 强度归一化 |
| `histogram_standardization` | 直方图标准化 |
| `adaptive_histogram_equalization` | 自适应直方图均衡 |
| `custom_preprocessor` | 自定义模板 |

`load_image` 是惰性注册：它有注册装饰器，但未被 `habit/core/preprocessing/__init__.py` 导入，需经 `image_processor_pipeline` 加载后才出现在 `available()` 中。

### 11.2 体素/聚类特征 `domain="feature_extractors"`（v0.1.x 遗留域）

| 名称 | 说明 | v1.0 归属 |
|---|---|---|
| `raw` | 原始体素强度 | `voxel_feature_extractor` |
| `kinetic` | 动态增强动力学特征 | 尚未迁入 v1.0 域 |
| `local_entropy` | 局部熵 | 尚未迁入 v1.0 域 |
| `voxel_radiomics` | 逐体素影像组学 | 尚未迁入 v1.0 域 |
| `supervoxel_radiomics` | 逐超体素影像组学 | `supervoxel_feature_extractor` |
| `mean_voxel_features` | 体素特征均值聚合 | `supervoxel_feature_extractor` |
| `concat` | 多提取器拼接 | 尚未迁入 v1.0 域 |
| `my_feature_extractor` | 示例实现，**默认已加载** | 仅遗留域 |
| `custom_template` | 自定义模板，默认不加载 | 视实现的协议而定 |

🔴 注意 v1.0 的 `voxel_feature_extractor` 域目前**只有 `raw`**，`kinetic` / `local_entropy` / `voxel_radiomics` 还留在遗留域里，尚未接入 `VoxelFeatureExtractor` 协议。

### 11.3 聚类算法 ⚠ v0.1.x 无插件域，v1.0 拆为两个域

| 名称 | 说明 | v1.0 归属 |
|---|---|---|
| `kmeans` | K-means（支持 elbow / silhouette 选 k） | `habitat_model_fitter`，同名实现也在 `supervoxelizer` |
| `gmm` | 高斯混合（支持 AIC / BIC 选 k） | `habitat_model_fitter`，同名实现也在 `supervoxelizer` |
| `slic` | SLIC 超体素 | `supervoxelizer` |
| `nearest_centroid` | 用已 fit 模型给个体贴标签 | `habitat_assigner` |
| `custom_template` | 自定义模板，默认不加载 | 视实现的协议而定 |

v0.1.x 把它们塞进同一个 `ClusteringAlgorithmFactory`，但做的不是同一件事：`supervoxelizer` 在**一个个体内**划分区域，`habitat_model_fitter` 在**整个队列上**求共享的生境定义，`habitat_assigner` 把定义套回个体。v1.0 按协议拆开，正是让"个体级 / 群体级"这条边界在注册表里也显式化。`kmeans` / `gmm` 在超体素域与生境模型域各有一份实现，同名但不同协议，靠注册表本身区分。

### 11.3b 聚类输入预处理 ⚠ v0.1.x 无插件域，v1.0 域名 `feature_preprocessing_method`

作用于**体素 / 超体素特征矩阵**（一行一个单元），在特征提取之后、聚类之前。

| 名称 | 说明 | 有状态可学到什么 |
|---|---|---|
| `impute` | 非有限值填充（`strategy`: `mean` / `median` / `zero`） | 每列填充值 |
| `minmax` | 缩放到 `[0, 1]` | 每列或全局的 min/max |
| `zscore` | 标准化 | 每列或全局的 mean/std |
| `robust` | 中位数 / IQR 标准化 | 每列或全局的 median/IQR |
| `log` | `log1p`（先平移到非负） | 平移量 |
| `winsorize` | 按分位数截尾（`winsor_limits`） | 上下界 |
| `binning` | 离散成序数箱号（`n_bins`、`bin_strategy`: `uniform` / `quantile` / `kmeans`） | 箱边界 |
| `variance_filter` | 丢弃低方差列（**至少保留一列**） | 保留列名 |
| `correlation_filter` | 丢弃高相关列中的后者 | 保留列名 |

要点：

- 前七个是**逐列数值变换**，后两个是**列筛选**。同一套方法既服务无状态的 `SubjectPreprocessingChain`，也服务有状态的 `CohortPreprocessingChain`——方法不需要知道持有它的链会丢弃还是保存它的状态。
- `binning` 是 radiomics 特征集最典型的队列级步骤：把连续值换成箱号，丢掉主要反映采集噪声的细微变化，保留承载生物学信息的次序。箱边界来自汇总后的队列，因此同一个箱号在不同受试者之间含义相同——放在个体级链里就没有这个性质。
- `across_features`（v0.1 叫 `global_normalize`，`legacy.py` 自动转换）决定统计量在**特征列之间**汇总还是每列各算一份。对多模态特征这是科学选择：跨列汇总保留模态之间的相对强度尺度，逐列缩放会抹平它。
- `impute` 在未显式配置时由链**自动前置**（`strategy="mean"`，兼容 v0.1），并写进 `spec` 以保证溯源里没有"记录之外的计算"。
- `variance_filter` 永不返回空矩阵：全部列都低方差时保留方差最大的一列。空矩阵会让下游聚类以一个无法解释的错误崩掉，而这通常意味着上游参数错了而非数据真的没有信息。

### 11.4 特征表预处理 ⚠ v0.1.x 无插件域，v1.0 域名 `table_preprocessor`

作用于**建模表**（一行一个受试者，带 ID 与结局列），在特征选择与建模之前。

`minmax`、`zscore`、`robust`、`binning`、`winsorize`、`log`、`variance_filter`、`correlation_filter`、`custom_template`（模板，默认不加载）

与 §11.3b 同名的方法**共用同一份数值实现**（L0 `habit.kernels.feature_transforms`），但**域不同**：那边的一行是一个体素或超体素（通往生境定义），这边的一行是一个受试者（通往结局模型）。行的语义不同，能做的检查也不同（这边要认 ID 列与结局列），所以是两个域而不是一个域两种用法。

参数名两个域完全一致，包括 `across_features`——同一个 v1 版本里同一个概念不该有两个名字。

### 11.5 生境特征 `domain="habitat_feature_extractor"`（遗留别名 `"habitat_features"`）

| 名称 | 说明 |
|---|---|
| `traditional` | 生境区域的传统影像组学特征 |
| `non_radiomics` | 非组学特征（体积、数目、占比等） |
| `whole_habitat` | 整体生境层面特征 |
| `each_habitat` | 每个生境分别提特征 |
| `msi` | Multi-region Spatial Interaction，空间交互指标 |
| `ith_score` | 瘤内异质性评分 |
| `volume` | 各生境体素数与体积占比（v1.0 新增，**只在 `habitat_feature_extractor` 域**） |
| `custom_foreground_volume` | 自定义模板，默认不加载 |

内置插件在 import `habit.domain` 时自动注册；模板需手动 import。遗留别名域 `habitat_features` 只含前 6 个，不含 `volume`。

### 11.6 机器学习模型 `domain="classifier"`（遗留别名 `"models"`）

`LogisticRegression`、`SVM`、`SVC`、`RandomForest`、`XGBoost`、`DecisionTree`、`GradientBoosting`、`AdaBoost`、`MLP`、`KNN`、`GaussianNB`、`MultinomialNB`、`BernoulliNB`、`AutoGluonTabular`（14 个生产模型）。示例实现 `CustomEnsemble` 只在遗留域 `models` 里。

### 11.7 特征选择 ⚠ v0.1.x 无插件域，v1.0 域名 `feature_selector`

| 名称 | 说明 |
|---|---|
| `variance` | 方差阈值（默认在 z-score 前执行） |
| `correlation` | 相关性过滤 |
| `icc` | 按 ICC 稳定性筛选 |
| `mrmr` | 最大相关最小冗余 |
| `lasso` | LASSO |
| `chi2` | 卡方检验 |
| `anova` | 方差分析 |
| `rfecv` | 交叉验证递归特征消除 |
| `vif` | 方差膨胀因子 |
| `statistical_test` | 组间统计检验 |
| `univariate_logistic` | 单因素 logistic |
| `stepwise` | 逐步回归 |

### 11.8 评估指标 `domain="metric"`（遗留别名 `"metrics"`）

`accuracy`、`sensitivity`、`specificity`、`ppv`、`npv`、`f1_score`、`auc`、`hosmer_lemeshow_p_value`、`spiegelhalter_z_p_value`

---

## 12. 扩展开发：写自己的组件

### 12.1 在自己的项目里注册（无需改 HABIT 源码）

注册写法与 v0.1.x 完全一致：`@<Registry>.register("name")`。v1.0 没有引入新的注册机制，只是把注册表补全并统一到同一套基类上。

个体级组件（实现 `__call__`，别名一行带过）：

```python
from habit.domain import VoxelFeatureExtractorRegistry
from habit.contracts import Subject, VoxelFeatureField

@VoxelFeatureExtractorRegistry.register("foundation_encoder")
class FoundationEncoderVoxelFeatures:
    """Per-voxel embeddings produced by a third-party image encoder."""

    def __init__(self, checkpoint: str, layer: int = -2) -> None:
        self.checkpoint = checkpoint
        self.layer = layer

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """
        Compute per-voxel embeddings for one subject.

        Args:
            subject: Subject providing the images and the ROI mask.

        Returns:
            One embedding vector per voxel inside the ROI.
        """
        ...

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification, used for provenance and caching."""
        return Spec(name="foundation_encoder",
                    params={"checkpoint": self.checkpoint, "layer": self.layer})
```

除 `__call__` 外，八个领域协议都要求一个 `spec` 属性——溯源与缓存键都从它来，缺了就不满足协议。实现了这两者，组件即可直接被 `cohort.map()`、`SubjectPipeline`、任何执行后端以及 MONAI 的 `Compose` 使用，不需要额外写适配器。

群体级组件（实现 `fit`）：

```python
from typing import Optional, Sequence

import numpy as np
from habit.domain import HabitatModelFitterRegistry
from habit.contracts import Cohort, HabitatModel, Supervoxelization

@HabitatModelFitterRegistry.register("my_spectral")
class MySpectralHabitatFitter:
    """Population-level habitat fitter based on spectral clustering."""

    def __init__(self, n_habitats: int = 4) -> None:
        self.n_habitats = n_habitats
        self._seed = 0

    def set_random_state(self, seed: int) -> None:
        """Set the seed used for spectral embedding initialisation."""
        self._seed = seed

    def fit(
        self,
        units: Sequence[Supervoxelization],
        *,
        cohort: Optional[Cohort] = None,
    ) -> HabitatModel:
        """
        Fit habitat definitions on the pooled supervoxel features of a cohort.

        Args:
            units: One supervoxelization per subject, in cohort order. Order is
                part of the contract because clustering can be order-sensitive.
            cohort: Cohort the units came from, recorded as a non-identifiable
                fingerprint inside the model.

        Returns:
            A self-describing habitat model usable on any new cohort.
        """
        matrix: np.ndarray = np.vstack([unit.features for unit in units])
        ...
        return HabitatModel(...)
```

参数名跟着协议走：形参是 `units` 而不是 `items`，且必须接受关键字参数 `cohort`；构造参数用 `n_habitats`（与内置 fitter 一致），不要另造 `n_clusters`。

注册后立即可用：

```yaml
# YAML side: the key is the slot in the habitat spec, `name` is the implementation
habitat_model:
  name: my_spectral
  params: {n_habitats: 5}
```
```python
# API side
fitter = HabitatModelFitterRegistry.create("my_spectral", n_habitats=5)
model = fitter.fit(units, cohort=cohort)
```

### 12.2 通过 entry point 做成可安装插件

```toml
# your_plugin/pyproject.toml
[project.entry-points."habit.habitat_model_fitter"]
my_spectral = "your_plugin.fitters:MySpectralHabitatFitter"
```
`pip install your-plugin` 后 `load_plugins()` 自动发现，CLI / API / GUI 三处同时可见。

🟢 分组名规则是 **`habit.` + 该注册表的域名**（`ComponentRegistry.entry_point_group()` 即 `f"habit.{domain}"`），域名取值见 §11 开头的对照表——v1.0 新增域是单数，v0.1.x 遗留域是复数，不存在"一律复数"的规则。当前可注册的分组：

| entry point 分组 | 对应注册表 |
|---|---|
| `habit.voxel_feature_extractor` | `VoxelFeatureExtractorRegistry` |
| `habit.feature_preprocessing_method` | `FeaturePreprocessingMethodRegistry` |
| `habit.supervoxelizer` | `SupervoxelizerRegistry` |
| `habit.supervoxel_feature_extractor` | `SupervoxelFeatureExtractorRegistry` |
| `habit.habitat_model_fitter` | `HabitatModelFitterRegistry` |
| `habit.habitat_assigner` | `HabitatAssignerRegistry` |
| `habit.habitat_feature_extractor` | `HabitatFeatureExtractorRegistry` |
| `habit.table_preprocessor` | `TablePreprocessorRegistry` |
| `habit.feature_selector` | `FeatureSelectorRegistry` |
| `habit.classifier` | `ClassifierRegistry` |
| `habit.metric` | `MetricRegistry` |
| `habit.preprocessors` / `habit.feature_extractors` / `habit.habitat_features` / `habit.models` / `habit.metrics` | v0.1.x 遗留分组，继续可用 |
| `habit.radiomics_backends` | 占位，尚未 registry 化 |

v0.1.x 那几个装不进外部插件的注册表（聚类算法、聚类输入预处理、特征表预处理、特征选择）已在 v1.0 补齐，对应上表的 `habit.supervoxelizer` / `habit.habitat_model_fitter`、`habit.feature_preprocessing_method`、`habit.table_preprocessor`、`habit.feature_selector`。

### 12.3 自定义数据来源

只要实现 `DataSource` 协议就能接任何存储：

```python
from habit.contracts import Cohort

class PacsDataSource:
    """Build a cohort directly from a PACS query result."""

    def load(self) -> Cohort:
        """Return the cohort described by this query, in a reproducible order."""
        ...
```

🟢 协议要求的方法名是 **`load()`**（内置实现 `DirectoryDataSource`、`NnUNetDataSource` 亦然），不是 `cohort()`。

---

## 13. 从 v0.1.x 迁移

| 你原来的做法 | v1.0 怎么办 |
|---|---|
| `habit get-habitat -c my.yaml` | **不用改**，照跑 |
| 旧 YAML 配置 | **不用改**；想用新字段就 `habit migrate-config -c my.yaml` |
| `habitat_pipeline.pkl` | 可直接加载；新产物为 `.habitatmodel` |
| `from habit import run_habitat_analysis` | 仍可用（发 `DeprecationWarning`），建议改 `habit.recipes.two_step_habitat` 或 `run_from_yaml` |
| `from habit import Cohort` | 仍可用，但语义升级为真正的数据对象而非目录包装 |
| 依赖 `_PIPELINE_RECIPES` 改源码加范式 | 改为实现协议 + 注册，不再需要 fork |
| 依赖非商业许可条款 | 已改为 Apache-2.0，商业使用无需授权；请按 `CITATION.cff` 引用 |

行为差异（属于"允许改善"范围，不影响科学结果）：输出目录结构、日志格式、进度条呈现。

---

## 14. 附录：模块索引

| 模块 | 一句话 |
|---|---|
| `habit` | 稳定扁平 API 门面，惰性导入 |
| `habit.recipes` | 每个 CLI 命令对应一个配方函数（🔴 尚未落地） |
| `habit.contracts` | 领域数据模型（Subject / Cohort / HabitatModel / FeatureTable / Provenance …） |
| `habit.domain` | 八个领域协议与内置实现，全部单个体可直接调用 |
| `habit.kernels` | 纯数值计算，无 IO 无状态。MSI / ITH 等有科学定义的公式为稳定公开 API，其余内部 |
| `habit.adapters` | 数据来源与产物落地（目录 / DataFrame / 内存 / nnU-Net） |
| `habit.execution` | 执行后端与检查点 |
| `habit.registry` | 注册表基类与 entry point 加载；自省函数 `list_plugins` / `get_plugin_info` / `get_param_schema` / `load_plugins` 从顶层导出 |
| `habit.spec` | Spec / RunPolicy / YAML 双向同构 / 旧配置翻译 |
| `habit.compat.sklearn` | 与 sklearn Pipeline / GridSearchCV 互通 |
| `habit.compat.monai` | 与 MONAI Dataset / transform 互通 |
| `habit.compat.nnunet` | 直读 nnU-Net 数据集 |
| `habit.utils` | 统一工具：日志、进度条、可用性探测 |
| `habit.cli` / `habit.commands` | 命令行入口，仅解析与装配 |
