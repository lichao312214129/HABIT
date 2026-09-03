# 09 — 物理能力包与公开 API 对齐重构计划

> 状态：**已定案，目标版本 v2.0.0，待实施**（2026-09-02）。
>
> 本文件增补 `08_naming_decisions.md` §5。`habit.domain` 不再是 L3 算法与组件的
> 物理家园；L3 不再以一个用户不可理解的 `domain` 包表达，而由按能力命名的公共包组成。
> 在 v2.0.0 中 `habit.domain` 和旧的扁平组件导入一并删除。与旧的 09 facade 方案冲突处，
> 以本文为准。

---

## 1. 决策与动机

### 1.1 要解决的真实问题

以下两行在同一张 “Voxel feature extractors” API 表内出现：

```python
habit.extract_voxel_texture(image, mask)
habit.domain.RawVoxelFeatures(modalities)
```

两者同属一项能力，却有不同的导入层级。`domain` 还是内部架构的 L3 术语，而非用户的
任务词汇。这会造成三类长期问题：

1. 用户无法判断哪个是规范 API，文档无法只教一种写法；
2. 新组件没有确定落点，容易被随意提升到包根或塞进 `domain`；
3. 新增 `habit.<capability>` facade、却让实现仍住在 `habit.domain`，形成双重组织中心；
   Sphinx 的 duplicate-object 警告已经证明这不是可维护状态。

### 1.2 最终原则

1. **物理真身 = 规范公开路径。** `habit.voxel_features.RawVoxelFeatures` 的类定义、
   Registry、Params、树构建器均在 `habit/voxel_features/`，不从 `habit.domain` 导入。
2. **用户路径按能力命名。** 新的用户文档、示例、测试全部使用
   `habit.<capability>`；不再教 `habit.domain.*`。
3. **删除旧路径。** v2.0.0 删除 `habit.domain`、其历史深层模块路径和旧的扁平组件别名；
   不保留 shim、re-export 或双路径文档。
4. **不改科学定义。** 只移动文件与 import；公式、默认值、Registry domain 名、
   entry-point 组、`Spec` 名/参数、随机种子、数值归约与 golden 容差均不变。
5. **一个符号一个规范路径。** `habit.__init__` 只保留版本与包发现，不再镜像组件；
   规范路径只显示 `habit.<capability>.X`。

### 1.3 构造器是唯一参数契约（v2）

组件不再定义或公开 `*Params` Pydantic 配套模型，也不再在 Registry 保存
`params_model` 作为第二份 schema。每个组件的 `__init__` 是参数的唯一真相来源，
采用 sklearn 的可检查约定：

1. 每一个可配置参数都有完整类型标注、默认值；有限候选使用 `Literal` 或 `Enum`，
   数值边界用 `Annotated`/显式运行期校验表达；
2. 构造器执行输入验证，并把每个参数保存为同名实例属性（或 sklearn 兼容属性）；
3. Registry 只负责 `name → class` 与 `create(name, **kwargs)`，不做参数 schema
   注册、Pydantic 验证或双重维护；
4. 参数目录、GUI/API 自省和 Sphinx 文档从 `inspect.signature`、类型注解及构造器
   docstring 的 `Parameters` 段生成。每个组件页展示调用构造器及其参数的类型、默认值、
   允许值/范围和语义，不生成 `FooParams` 页面；
5. `Spec.params` 继续是无类型 `dict`，只作为构造器 kwargs 与可复现溯源载体，不依赖
   Pydantic 参数模型。

这同样是 v2 的破坏性变更；不保留 Params 导入、`get_params_model()` 或旧 YAML
参数 schema compatibility。迁移不得改变构造器的科学默认值或公式。

### 1.4 成熟库对照

| 库 | 规范用户路径 | HABIT 采用的结论 |
|---|---|---|
| scikit-learn | `sklearn.svm.LinearSVC`、`sklearn.feature_selection.SelectKBest` | 按能力子包公开；测试按用户路径导入 |
| MONAI | `monai.transforms.Compose`、`monai.metrics.*` | 变换、指标、数据等能力独立命名空间 |
| PyRadiomics | `radiomics.featureextractor.RadiomicsFeatureExtractor` | 提取能力在语义子包，不暴露内部架构层 |

不采用“把所有组件扁平导出到 `habit.X`”的做法：包根会成为数百个名称的无结构字典，
既不利于发现，也不符合上述库的模式。

---

## 2. 目标物理包结构

```text
habit/
├── voxel_features/          # VoxelFeatureExtractor、所有体素特征、Registry、voxel tree
├── supervoxel/              # 划分 + 超体素特征 + 两个 Registry、supervoxel tree
├── feature_preprocessing/   # 聚类输入预处理（subject/cohort）
├── habitat_model/           # fitter + assigner + label postprocess
├── habitat_features/        # 生境级特征、比较、habitat tree
├── combiners/               # 组合器
├── pipeline/                # SubjectPipeline/TablePipeline、stages、pooling、assembly
├── table_preprocessing/     # 一行一受试者的表预处理
├── feature_selection/       # 表特征选择
├── classification/          # 分类器
├── regression/              # 回归器
├── survival/                # 生存模型
├── evaluation/              # 指标、统计、split、outcome access
├── image_preprocessing/     # 图像预处理与几何对齐
├── precision/               # 扰动、ICC screen、label stability
├── radiomics/_domain.py     # voxel/supervoxel/habitat/precision 共用的内部 radiomics 基元
```

`contracts`、`kernels`、`adapters`、`execution`、`registry`、`spec`、`recipes`、`report`、
`viz` 维持现有职责和位置。它们已经是可理解的能力/基础概念，不在本次改名范围。

### 2.1 v2 架构边界（从既有设计吸收）

依赖只允许向下：`L0 kernels → L1 adapters → L2 contracts → L3 capability
packages → L4 recipes → L5 CLI/GUI`。L0–L3 不得依赖 YAML、目录约定或输出路径；
除 adapters 和显式 `StudyResult.save()` 外不得写文件。单受试者算子必须能够接收内存
`Subject` 并返回内存结果；队列、并行和持久化是外层能力。每个可插拔组件继续携带
`Spec`，每个结果继续携带 `Provenance`，且群体预处理状态必须随 `HabitatModel` 保存。

### 2.2 当前 → 目标映射

| 当前真身 | 迁至规范物理包 | 同包原因 |
|---|---|---|
| `domain/voxel_features/`、`trees.VoxelFeatureTree` | `voxel_features/` | 每个 ROI voxel 的描述与组合 |
| `domain/supervoxel/`、`supervoxel_features/`、supervoxel tree | `supervoxel/` | 划分及其区域描述是同一步骤 |
| `domain/feature_preprocessing/` | `feature_preprocessing/` | 聚类输入的矩阵预处理 |
| `domain/habitat_model/`、`assignment/`、`postprocess/` | `habitat_model/` | 拟合、映射、标签清理共同构成模型生命周期 |
| `domain/habitat_features/`、habitat tree | `habitat_features/` | 对已得生境图定量 |
| `domain/combiners/` | `combiners/` | 三个粒度共享的列向组合 |
| `domain/pipeline.py`、`stages/`、`pooling*`、`assembly.py`、`sklearn_interop.py` | `pipeline/` | 组件图装配与执行的数据流 |
| `domain/table_preprocessing/` | `table_preprocessing/` | 建模表预处理，避免与聚类输入混淆 |
| `domain/feature_selection/` | `feature_selection/` | 与 sklearn 同词 |
| `domain/classification/`、`regression/`、`survival/` | 同名包 | 不混淆终点建模任务 |
| `domain/evaluation/`、`split.py`、`outcome_access.py` | `evaluation/` | 指标、统计与数据划分 |
| `domain/image_preprocessing/`、`geometry_align.py` | `image_preprocessing/` | 图像变换及几何一致性 |
| `domain/precision/` | `precision/` | 可重复性/可复现性分析 |

### 2.3 共享抽象的物理归属

| 当前模块 | 目标 | 要求 |
|---|---|---|
| `domain/protocols.py` / `table_protocols.py` | 各能力包的 `protocols.py`；`Seedable` 仅保留为共享基础协议 | `VoxelFeatureExtractor` 不再住在 `domain` |
| `domain/trees.py` | 各粒度包的 `trees.py` | 不复制树逻辑；先抽共享无状态 helpers |
| `domain/habitat_features/_radiomics.py` | `radiomics/_domain.py` | 先于 voxel/supervoxel/habitat/precision 迁移，避免横向循环依赖 |
| `domain/assembly.py` | `pipeline/assembly.py` | 保持 `Spec → components` 单一装配点 |
| `domain/stages/`、`pooling.py`、`pooling_marker/` | `pipeline/` | 保持 subject→cohort 分水岭语义 |
| `domain/geometry_align.py` | `image_preprocessing/geometry.py` | 只移动几何实现，不改变 resampling 定义 |

---

## 3. v2 迁移与产物策略

### 3.1 导入与插件

- `habit.<capability>.X` 是唯一 canonical import。
- `habit.domain.X`、历史 `habit.domain.<submodule>.<file>.X` 和旧的扁平组件别名全部删除；
  升级 v2 前的调用方必须按发布说明迁移。
- Registry `domain`（如 `voxel_feature_extractor`）、entry-point 组
  `habit.voxel_feature_extractor`、`Spec.name`、YAML 字段均保持冻结；**Python 包移动不能改变
  插件发现规则**。
- `habit.api.plugins._V1_DOMAIN_REGISTRIES` 等硬编码表改为新 canonical 路径；插件输出显示
  新 `type.__module__`。

### 3.2 产物与序列化

| 产物 | 风险 | 必须措施 |
|---|---|---|
| `.habitatmodel` | 基于 registry name / `Spec`，风险较低 | 加载新旧模型并对比 `model_id`、标签图和特征 |
| `.habitpipeline` | payload 直接 pickle，记录类 module path | 旧文件明确拒绝并抛出 `CompatibilityError`，提示固定到 v1 或按迁移工具重建；不可静默加载 |
| Run manifest / `PluginInfo.implementation` | `type.__module__` 将改为新路径 | 新写入 canonical 新路径；旧记录仅作为历史文本读取，不尝试复建类 |
| 外部插件 | 用户插件可能 import `habit.domain.*` | v2 明确要求改 import；entry-point 分组不改 |

### 3.3 API 注册表与文档

`_public_api.py` 拆为两种不混淆的声明：

```python
# Package root exposes only package discovery and version metadata.
PUBLIC_API_SYMBOLS: tuple[str, ...]

# New: each public capability package and its canonical symbols.
PUBLIC_NAMESPACES: dict[str, tuple[str, ...]]
```

强制不变量：

1. `habit.<capability>.__all__` 等于 `PUBLIC_NAMESPACES`；
2. capability 导出的对象就是其物理模块定义的对象；
3. 一个符号只能在一个 capability namespace 中作为 canonical；
4. `habit.domain` 不可导入，旧扁平组件名不可解析；
5. `import habit` 不加载 radiomics、sklearn 等重依赖。

autosummary 每个对象只为 canonical 路径生成一个 stub；所有旧
`api/generated/habit.domain.*` stub 必须删除，防止 duplicate object / orphan warnings。

---

## 4. 迁移执行顺序

原则：**先抽共享依赖，再迁叶包，再迁 pipeline 枢纽，最后删除 `domain`**。全程 `git mv`，
禁止复制算法到新旧两处。

### 阶段 0 · 基线与迁移基础

1. 恢复当前薄 `habit.voxel_features` 试验的文档/API 变动，避免中间态被当作发布 API；
2. 固化并测试 `_radiomics`、几何对齐、`Seedable` 的目标边界；
3. 写旧 `.habitpipeline` 的版本检测与明确拒绝策略，禁止因 pickle 缺模块而给出含混的
   `ModuleNotFoundError`；
4. 架构门禁从“只检查 `habit.domain`”改为可声明每个 L3 capability package 的禁入表模板。

### 阶段 1 · 独立叶包

依次 `git mv`：`classification`、`table_preprocessing`、`feature_selection`、`evaluation`、
`image_preprocessing`、`regression`、`survival`。每个包是一个独立、可回滚的提交；更新生产
import、测试和公开 API 表。先修复 regression / survival 当前未进入统一自省目录的不一致。

### 阶段 2 · Voxel 真身试点

1. `git mv habit/domain/voxel_features/* → habit/voxel_features/`，将现有薄
   `habit/voxel_features/__init__.py` 改为真实聚合；
2. 拆 `VoxelFeatureTree` / `build_voxel_extractor`，协议迁至该包；
3. 删除 `habit/domain/voxel_features/*`；生产 import、文档、autosummary 统一新路径；
4. 验收 API 页面全部显示 `habit.voxel_features.*`，无重复对象。

### 阶段 3 · 组合器与超体素

先迁 `combiners` 并消解其对 expression/kinetic voxel 实现的依赖；再迁
`supervoxel` + `supervoxel_features` 及树，合并为 `habit.supervoxel`。

### 阶段 4 · Habitat 核心

迁 `habitat_model` + `assignment` + `postprocess`，再迁 `habitat_features`。
此阶段必须验证 `HabitatModel` 加载、assiger 重建、逐体素标签与方法学报告。

### 阶段 5 · 预处理、精筛、pipeline

迁 `feature_preprocessing` 与 `precision`；最后迁 `pipeline.py`、`stages/`、`pooling*`、
`assembly.py`、`sklearn_interop.py`。pipeline 是所有注册表的汇点，只能在叶包稳定后处理。

### 阶段 6 · 删除 `habit.domain` 与旧扁平组件别名

1. 所有生产代码、文档、示例、测试中的 `habit.domain` import 归零；
2. 删除 `habit/domain/`、旧 domain autosummary stubs 与顶层组件别名；
3. 添加旧 `.habitpipeline` 的明确版本拒绝测试；
4. 发布 v2 迁移说明和破坏性变更列表。

---

## 5. 每阶段门禁与完成条件

每个 capability 包迁移后必须依次通过：

```powershell
& "E:\conda\mconda\envs\py310\python.exe" -m pytest tests/test_architecture_contracts.py
& "E:\conda\mconda\envs\py310\python.exe" -m pytest tests/api/test_public_api.py tests/api/test_docs_registry_examples.py
& "E:\conda\mconda\envs\py310\python.exe" -m pytest tests/golden/fast
```

此外：

- 运行该包对应的原有 domain/registry/recipe 测试，不通过修改期望掩盖回归；
- 按文档的 import 与最短真实示例运行（voxel 首阶段至少运行
  `custom_voxel_feature_demo.py`，`HABIT_NO_VIEW=1`）；
- Sphinx 必须是 **0 errors、0 warnings**；
- 触及标签或特征的阶段，在本地 `demo_data/` 运行 full baseline，标签逐体素一致、特征
  `rtol=1e-6`；
- 新包自身、`registry`、`spec`、`contracts` 不得反向 import recipes/CLI/compat 引擎。

---

## 6. 审查清单与非目标

### 审查清单

- [ ] 没有算法被复制；`git diff` 仅允许移动、import、注册表、文档和测试变化。
- [ ] `Registry.create` / `Spec` / entry-point 名称和默认值未变。
- [ ] 每个组件只有一个 canonical import；旧 `habit.domain` 与旧扁平组件名均不可导入。
- [ ] v1 `.habitpipeline` 被明确拒绝；`.habitatmodel` 按格式版本加载或明确拒绝。
- [ ] `PluginInfo.implementation` 的新路径准确，未将旧产物伪写成新路径。
- [ ] 不存在 `habit/domain/` 或 `habit.domain` production import。
- [ ] 真实文档示例和 Sphinx 均通过。

### 非目标

- 不在这次迁移中改变任何统计定义、IBSI 设置、随机过程、默认超参数或结果文件值；
- 不改 Registry `domain`/entry-point 分组；
- 不为 v1.x 保留 `habit.domain` 或顶层旧组件导入；这是 v2.0.0 的明确破坏性变更；
- 不保留组件 `*Params` Pydantic 模型或 Registry `params_model` 双重 schema；
- 不把 `kernels` / `viz` 顶层镜像策略一并重开；该讨论留给 v2。
