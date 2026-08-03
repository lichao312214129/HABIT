# 06 — HABIT v1.0 架构设计：API 优先

> **状态**：设计评审中（分支 `v1.0.0`）
> **取代**：`01_master_plan.md` 与 `02_public_api_design.md` 中"API 是 CLI 门面、不重构业务逻辑"的定位。
> 那两份文件描述的 A 档 MVP 已完成并随 v0.1.x 发布，其内容作为**历史记录**保留，不再作为 v1.0 的执行依据。

---

## 1. 定位变更

| | v0.1.x（现状） | v1.0（本设计） |
|---|---|---|
| **系统是什么** | 配置驱动的应用程序，附带一个 API 门面 | 一个 Python 科学库，附带一个配置驱动的 CLI 壳 |
| **执行单元** | 整条工作流（`run_habitat_analysis`） | 领域算子（`Supervoxelizer`、`HabitatModelEstimator` …） |
| **数据契约** | 文件系统目录约定 | 内存对象（`Subject` / `Cohort` / `HabitatModel` …） |
| **依赖方向** | CLI → config → 核心（核心读 config 字段） | 核心 ← 适配层 ← CLI（核心不知道 YAML 存在） |
| **核心产物** | 磁盘上的一堆 CSV 与 `habitat_pipeline.pkl` | `HabitatModel` —— 可检视、可分享、可引用的科学产物 |
| **扩展方式** | 改 HABIT 源码（如 `_PIPELINE_RECIPES`） | 实现协议 + 注册，无需改源码 |

**一句话**：v0.1 把配置当作架构；v1.0 把配置当作**界面**，架构交给领域对象。

---

## 2. 目标与非目标

### 2.1 目标

| 编号 | 目标 | 可验证标准 |
|---|---|---|
| **G1** | **嵌入生态**（首要目标） | 第三方能在自己的流水线中单独调用 HABIT 的任意一环，输入输出均为内存对象，不需要接受 HABIT 的目录约定 |
| **G2** | **领域词汇即 API** | 影像组学研究者读 API 名称即能理解其科学含义，无需先读文档 |
| **G3** | **`HabitatModel` 可流通** | 一篇论文发布的生境模型，他人可加载并在自己队列上复现同一套生境定义 |
| **G4** | **溯源与报告内建** | 跑完一次分析可自动导出 Methods 段落与 CLEAR / IBSI 条目自查表 |
| **G5** | **非编程用户无感** | 现有 CLI 命令与旧 YAML 继续可用，普通用户的操作方式不变 |

### 2.2 非目标

- **不改变生境分析的科学定义**。MSI、ITH score、non_radiomics 的公式与文献依据保持不变，重构只改变它们被调用的方式。
- **不重写 PyRadiomics / sklearn 已有能力**。HABIT 只做生境层的科学内涵。
- **不做 GUI 重构**。GUI 在新架构下自然成为又一个 driver，本设计只保证它可以这样接。
- **不追求一次性迁移全部子系统**。垂直切片优先（见第 12 节）。

---

## 3. 五个核心洞察

### 3.1 依赖倒置：核心不能知道 YAML 存在

当前 `HabitatAnalysis` 等核心类直接读 config 字段，导致"改配置格式 = 改核心"。v1.0 的铁律：

> **L0–L3 层的代码中不允许出现 `yaml`、`out_dir`、`config_file`、`run_mode`、`data_dir` 这些概念。**

配置的解析、路径解析、输出目录的组织，全部收敛到 L5 driver 层。

### 3.2 领域词汇即 API

通用抽象（`Operator.transform(x)`）虽然灵活，但对医学影像研究者不可读，也不利于第三方判断该扩展哪一处。v1.0 的核心协议数量刻意压到**五个**，且每一个都对应 `mental_model.rst` 里已有的领域词条：

```
Voxel feature  →  Supervoxel  →  Habitat  →  Habitat feature
```

这不是妥协，而是设计要求：**API 的可读性等于领域词汇的准确性。**

### 3.2b 单个体是原子调用

领域词汇解决"叫什么"，调用约定解决"怎么调"。第二个问题同样是生态问题：

```python
field = voxel_features(subject)      # 一个个体，不需要队列，不需要后端，不需要配置
```

每个个体级算子都是**单参可调用对象**，`Cohort` 与 `ExecutionBackend` 是叠在上面的可选设施。这是 MONAI（`Transform.__call__(data)`）、TorchIO（`transform(subject)`）、PyRadiomics（`extractor.execute(img, mask)`）共同收敛到的约定，对 HABIT 而言是**硬要求而非风格偏好**：只有单样本可调用对象才能被丢进 `monai.transforms.Compose`、被 torch `DataLoader` 驱动、或者在某一例结果异常时被单独拎出来调试。

同时每个协议提供**领域动词别名**（`extract` / `build` / `map`），实现方式是 `extract = __call__`，同一个函数两个名字，不存在两条代码路径。这样 `voxel_features(subject)` 与 `voxel_features.extract(subject)` 都成立，通用性与领域可读性不必二选一。

一个直接后果：`SubjectLevelOp` 不再引入第六个动词，它就是"带 `spec` 和 `cache_key` 的可调用对象"，所以四个个体级领域协议**自动满足**它，插件作者永远不需要为同一个计算写两个方法名。

### 3.3 `HabitatModel` 是可流通的科学产物

`two_step` 产出的群体级生境定义是 HABIT 最珍贵的东西——它让生境定义可以跨队列泛化，是外部验证与高质量发表的前提。现在它被压成一个 `habitat_pipeline.pkl` 黑箱。

v1.0 把它提升为一等对象：自带群体质心、特征定义、预处理状态、来源队列指纹、软件与随机种子指纹，可 `describe()`、可 `save()` / `load()`、可被论文引用。

> 类比：nnU-Net 的生态价值不只在代码，而在**预训练模型可以传播**。HABIT 对应的可传播物就是 `HabitatModel`。

### 3.4 个体级 / 群体级是架构骨架，必须显式化

这个两级结构目前隐含在 pipeline steps 的排列顺序里，但它同时是**四条边界**：

| 边界 | 含义 |
|---|---|
| 并行边界 | 个体级天然可并行；群体级必须等待汇总 |
| 断点续跑边界 | checkpoint 只在个体级有意义 |
| train / predict 边界 | 群体级 `fit` 产出模型，个体级套用模型 |
| **联邦边界**（未来） | 个体级在院内执行，只上传超体素特征；群体级聚类在中心完成 |

最后一条尤其重要：**`two_step` 的结构天然适合联邦生境建模**（影像不出院）。要保住这个潜力，两级结构必须成为显式契约。

### 3.5 溯源是数据结构的一部分，不是一个功能模块

`Provenance` 随每个数据对象传播，而不是在工作流末尾拼一份 manifest。这样任意一个中间产物都能回答"我是怎么来的"，第三方在自己的流水线中只用了 HABIT 的一环时，溯源信息依然完整。

**战略机会**：生境分析目前在国际上没有公认的报告规范。HABIT 有条件用可执行的方式去定义它。

---

## 4. 分层架构

```
L5  interfaces   CLI / YAML / GUI / REST      ← 解析与装配，无业务逻辑
L4  recipes      标准研究配方 + RunManifest 汇总 + 报告导出
L3  domain ops   五个领域协议 + Registry
L2  contracts    领域数据模型 + 两级算子协议 + 执行后端协议
L1  adapters     DataSource / Sink：目录约定、nnU-Net、MONAI、DataFrame、内存
L0  kernels      纯计算（numpy / SimpleITK / torch），无 IO、无状态、无日志
```

### 4.1 依赖规则

- 依赖**只能向下**，不允许反向或跨层向上。
- L0 不导入 L1 以上任何模块；L2 不导入 L3 以上任何模块，依此类推。
- L1 适配层是**唯一**允许触碰文件系统的层（L4 的 Sink 写出除外）。
- 该规则由架构契约测试强制（扩展现有 `tests/test_architecture_contracts.py`）。

### 4.2 与现有代码的对应

| 新层 | 现有代码的去向 |
|---|---|
| L0 | `clustering_features/`、`habitat_features/` 中的纯计算函数（MSI、ITH 等公式**原样保留**） |
| L1 | `utils/io_utils.get_image_and_mask_paths` → `DirectoryDataSource`；新增 nnU-Net / MONAI / DataFrame 适配 |
| L2 | 新建。部分复用现有 `habit/api/image.py` 的 `ImageVolume` / `MaskVolume` |
| L3 | `HabitatAnalysis` 拆解为五个协议的实现；八个 registry 收编统一 |
| L4 | `run_*_from_config` 的编排部分 + `RunManifest` |
| L5 | `habit/cli.py` + `habit/commands/` 基本不动，内部改调 L4 |

---

## 5. L2 数据契约

这是全盘的地基，必须先钉死。原型见 `prototype/contracts.py`。

| 类型 | 职责 | 关键不变量 |
|---|---|---|
| `Geometry` | spacing / origin / direction / shape | 所有空间对象共享同一 geometry 才能运算 |
| `ImageVolume` / `MaskVolume` | 带几何的影像与掩膜 | 不可变；与 geometry 强绑定 |
| `Subject` | 一个受试者：`{modality: 影像}` + `{roi: 掩膜}` + 元数据 | **影像字段惰性**，可持有 loader 而非已加载数组 |
| `Cohort` | `Subject` 的有序容器 | 可迭代 / 切片 / filter；`subject_id` 唯一 |
| `VoxelFeatureField` | ROI 内每体素一个特征向量 | 行数 = ROI 体素数；携带体素索引与 geometry |
| `Supervoxelization` | 个体内超体素划分 + 每个超体素的特征 | 超体素标签覆盖整个 ROI，无空洞 |
| `HabitatMap` | 生境标签图 | 标签取值来自某个 `HabitatModel`，记录 `model_id` |
| **`HabitatModel`** | **群体级生境定义** | 自足：脱离训练输出目录即可套用到新队列 |
| `FeatureTable` | 特征表 + 列语义 | 明确区分 ID 列 / 特征列 / 标签列 |
| `Provenance` | 该对象的来源 | 沿数据流传播，不丢失 |

### 5.1 `Subject` 的惰性设计

这是同时满足"小队列全内存可组合"与"大队列不爆内存"的关键。`Subject.images["T1"]` 返回的是一个 `ImageRef`，`.load()` 时才真正读盘或解码。因此：

- 上层算子代码**完全不需要区分**这两种场景
- 多进程执行时传递的是轻量引用而非数组
- 第三方可以自己实现 `ImageRef`（例如从 PACS、从 zarr、从 torch tensor）

### 5.2 `HabitatModel` 的内容

```
model_id            稳定标识（含 spec 指纹）
n_habitats          生境数
feature_names       群体级聚类所用特征的名称与顺序
centroids           群体质心
preprocessing_state 训练期学到的预处理状态（binning 边界、归一化参数等）
spec                产生它的完整算法规格（可导出 YAML）
cohort_fingerprint  来源队列描述：n、模态、来源、伦理与可分享性声明
provenance          软件版本、依赖版本、随机种子、时间戳
```

`describe()` 返回可读摘要；`save()` / `load()` 用**版本化的自描述格式**（不是裸 pickle），保证跨 HABIT 版本可读或给出明确的不兼容提示。

---

## 6. L3 领域协议

原型见 `prototype/protocols.py`。

```python
VoxelFeatureExtractor(spec)          __call__(Subject)             -> VoxelFeatureField
Supervoxelizer(spec)                 __call__(VoxelFeatureField)   -> Supervoxelization
HabitatModelEstimator(spec)          fit(Sequence[Supervoxelization]) -> HabitatModel   ← 群体级
HabitatMapper(model, spec)           __call__(Supervoxelization)   -> HabitatMap        ← 个体级
HabitatFeatureExtractor(spec)        __call__(Subject, HabitatMap) -> FeatureTable
```

四个个体级协议都是单参可调用（`HabitatFeatureExtractor` 的两个参数都是个体级的，故仍属个体级），各自另有领域动词别名 `extract` / `build` / `map`。

### 6.0 `HabitatMapper` 的模型在构造期注入

`HabitatMapper(model)` 而不是 `mapper.map(unit, model)`。两个理由：

1. mapper 因此成为普通单参可调用对象，与其余个体级算子同形，组合与执行时不需要任何特例绑定；
2. **没有模型就构造不出 mapper**，于是"未 fit 先 predict"从运行期错误变成不可表达的状态——train/predict 一致性由类型结构保证，而不是靠约定。

常用写法由 `HabitatModel.mapper()` 工厂提供：`labels = model.mapper()(unit)`。

### 6.0b `SubjectPipeline`：类型安全的 Compose

个体级链条（体素特征 → 超体素化 → 映射）合成为**一个**可调用对象。不直接复用泛型 `Compose` 的原因是 HABIT 各步的输入输出类型是异构的（`Subject → VoxelFeatureField → Supervoxelization → HabitatMap`），抹平成统一 dict 恰好会丢掉这套设计赖以自检的契约。

它的实际价值：**`HabitatModel` + `SubjectPipeline` 正好是外部验证需要分发的一对**——生境定义，以及套用这个定义的过程。

### 6.0c `SeedControl`：随机性的统一控制

生境分析对随机种子异常敏感（k-means / GMM 初始化、SLIC 播种，加上群体级聚类还对受试者顺序敏感）。v0.1 让每个组件各自发明 `random_state`，导致一次运行无法整体重播、也无法如实报告。参照 MONAI 的 `Randomizable`，随机组件实现 `set_random_state(seed)`；确定性组件不实现该协议，这本身就是有用的溯源信息。

### 6.1 三种 clustering_mode 如何表达

现在 `_PIPELINE_RECIPES` 是硬编码 dict，加第四种范式必须改源码。新架构下三种模式只是这五个协议的**不同装配方式**：

| 模式 | 装配 |
|---|---|
| `two_step` | 体素特征 → 超体素化 → 群体级 fit → 逐个体 map |
| `one_step` | 体素特征 → （跳过超体素）→ 个体级 fit + map |
| `direct_pooling` | 体素特征 → 直接汇总体素 → 群体级 fit → 逐个体 map |

三者作为**内置配方**保留在 L4，用户想要第四种自己组装即可，不需要改 HABIT。

### 6.2 registry

现有八个 registry（预处理、模型、聚类、聚类特征、特征表预处理、生境特征、特征选择、评估指标）收编到统一机制，并补齐两条能力：

- **自省**：沿用 v0.1.x 已有的 `list_plugins(domain)`、`get_plugin_info(name, domain)`、`get_param_schema(name, domain)`，不另造同义函数；v1.0 只补齐覆盖面并提供 JSON Schema 导出
- **entry points**：沿用现有 `habit.*` 分组，新增五个领域协议对应的分组

自省能力直接服务 Agent 场景：LLM 可据此自动构造合法的 spec。

---

## 7. 执行后端

**后端是可选加速器，不是前置条件。** 三层递进，任何一层都能独立完成工作：

```python
field  = voxel_features(subject)                              # 一个个体
fields = cohort.map(voxel_features)                           # 整队列，默认串行
fields = cohort.map(voxel_features, backend=ProcessPoolBackend(8))  # 显式要并行
```

不构造后端也能跑通全流程。这条不只是易用性——示例里若先造后端再干活，会让 notebook 用户误以为必须先搭基础设施，直接把他们推回 CLI。

```python
class ExecutionBackend(Protocol):
    def map(self, op: SubjectLevelOp, items, *, checkpoint=None) -> Iterator[Outcome]
```

算子只声明"我是个体级、可并行的"，**不自己管进程池**。现有的这些字段全部从 config 移出、收进后端：

`processes`、`cap_processes_to_gpu_pool`、`individual_subject_timeout_sec`、`individual_subject_graceful_shutdown_sec`、`individual_subject_spawn_timeout_sec`、`on_subject_failure`、`oom_backoff`、`oom_reduce_workers_by`、`resume`

内置后端：

| 后端 | 用途 |
|---|---|
| `SerialBackend` | notebook、调试、小队列 |
| `ProcessPoolBackend` | 迁移现有的超时 / OOM 退避 / 优雅关闭 / 单例失败隔离 |
| `CheckpointStore` | 断点续跑，作为后端的正交关注点 |

将来接 Dask / Ray / 集群调度只是多一个实现，**算法代码零改动**。

---

## 8. Spec 与 YAML 双向同构

### 8.1 三分

上帝对象 `HabitatAnalysisConfig` 拆成三个正交的东西：

| 对象 | 内容 | 性质 |
|---|---|---|
| `Spec` | 纯算法参数 | 可 hash、可 diff、无路径、进论文 Methods |
| `DataSource` | 数据从哪来 | 目录约定 / nnU-Net / MONAI / DataFrame / 内存 |
| `RunPolicy` | 后端、并发、超时、失败策略、OOM 退避、断点续跑、缓存目录 | 与科学结论无关；是执行参数的声明式快照，由适配层翻译成后端对象 |
| `ArtifactSink` | 产物写到哪、写不写 | 与科学结论无关，但与 `RunPolicy` 正交：内存运行时不给 sink 即可 |

判据很简单：**改了会影响科学结论的进 `Spec`，不影响的进 `RunPolicy`，与"数据在哪"有关的进 `DataSource`。**

### 8.2 双向同构（G5 的技术基础）

> **任何 Python 构造的分析必须能 `to_yaml()`；任何 YAML 必须能构造出等价的 Python 对象。**

这一条同时满足四类用户而不牺牲任何一类：

- 临床医生继续改 YAML（未来 GUI 生成 YAML），门槛为零
- 方法学家写 Python，写完 `spec.to_yaml()` 导出给临床同事复用
- Agent 生成哪种都行，且有 JSON Schema 可校验
- **论文补充材料直接附那份 YAML**，可复现性天然成立

### 8.3 指纹

`Spec.fingerprint()` 是对规范化后参数的稳定哈希，用于：`HabitatModel.model_id`、缓存键、checkpoint 键、以及判断"两次运行的算法是否真的一致"。

---

## 9. Provenance 与报告（v1.0 的差异化重点）

### 9.1 传播规则

每个算子在产出对象时，把「输入对象的 provenance + 自身 spec 指纹 + 环境指纹」合成新的 provenance。规则由基类统一实现，算子作者无需手写。

### 9.2 汇总与导出

L4 的 `RunManifest` 汇总一次分析的全部 provenance，并提供：

```python
result.manifest.describe_methods(style="radiology")  # 可直接进论文 Methods 的英文段落
result.manifest.checklist("CLEAR")                   # CLEAR / METRICS / IBSI / TRIPOD+AI 条目自查
result.manifest.to_json(path)                        # 机器可读，供 Agent 与 CI 使用
```

`describe_methods` 与 `HabitatSpec.describe_methods` 同名同签名，区别只在完整度：**spec 描述的是打算做什么**（跑之前就能看），**manifest 描述的是实际做了什么**（含版本、种子、失败的受试者）。同一个动词不引入第二个名字。

**Methods 文本必须只陈述实际执行过的步骤与参数**，不得生成未执行的内容——这是可信性的底线，需在实现中以测试保证。

---

## 10. 生态适配层

全部放进 `habit.compat.*`，可选依赖，不污染核心。

| 适配 | 内容 |
|---|---|
| `compat.sklearn` | 领域协议 → `BaseEstimator` 包装，可进 `Pipeline` / `GridSearchCV` |
| `compat.monai` | `Subject` ↔ MONAI dict 双向转换；HABIT 算子可直接作为 MONAI transform（前提是 3.2b 的单样本可调用约定），用户可继续用 torch `DataLoader` 做并行而不必交出执行控制权 |
| `compat.nnunet` | `NnUNetDataSource` 直读 `imagesTr/labelsTr` + `dataset.json` |

选择依据：这三者覆盖了影像组学研究者获得数据与做建模的主要路径。DICOM-SEG / BIDS 留作后续，接口上不设障碍（新增一个 `DataSource` / `Sink` 实现即可）。

---

## 11. CLI 与旧 YAML 兼容策略

### 11.1 三条保证

1. **CLI 命令名与选项不变**（`habit get-habitat -c xxx.yaml -m train` 等 15 个子命令全部保留）
2. **旧 YAML 继续可用**：冻结 v0 schema，由 `LegacyConfigAdapter` 翻译成 `(Spec, DataSource, RunPolicy)`
3. **提供 `habit migrate-config`**，把 v0 YAML 升级为 v1 格式；不升级也不影响使用

允许改善的部分（按你的判断"用户无感即可"）：输出目录结构、日志格式、进度条呈现。

### 11.2 "无感"的验收定义

用真实的 `config/` 模板 + `demo_data/` 跑端到端，验证：

- 命令退出码一致
- 产出的**科学结果文件**（特征 CSV、habitat map、模型指标）逐值一致
- 用户需要修改的 YAML 字段集合不变

---

## 12. 实施路线：垂直切片 = 生境分割

| 阶段 | 内容 | 验收 |
|---|---|---|
| **0** | **golden 基线固化**（在 `main` 上做） | demo 数据的生境标签、MSI、ITH、ML 指标固化为基线文件 |
| **1** | L2 契约层 + L1 目录 DataSource + `SerialBackend` | 契约测试通过；`import habit` 仍轻量 |
| **2** | **生境分割垂直切片**：五协议实现 + `HabitatModel` + `two_step` 配方 | 对着阶段 0 基线**逐值一致** |
| **3** | L5 适配：CLI 与旧 YAML 走新核心 | 第 11.2 节的无感验收全过 |
| **4** | `ProcessPoolBackend` 迁移（超时 / OOM / checkpoint） | 大队列跑通，行为与 v0.1 一致 |
| **5** | 特征提取 与 ML 子系统迁移 | 同样对基线验证 |
| **6** | `compat.*` 生态适配 + 报告导出 | 可运行的示例 |

阶段 2 是本次设计要验证的关键：**如果生境分割能在新架构下逐值复现，整套设计就成立。**

---

## 13. golden 基线策略（阶段 0）

| 项 | 做法 |
|---|---|
| 数据 | `demo_data/preprocessed/`（2 受试者 × 3 模态）+ `demo_data/ml_data/`（569 例 tabular） |
| 配置 | `config/habitat/config_habitat_two_step.yaml` 等现有模板，不改 |
| 锁定内容 | 生境标签图（逐体素）、`msi_features.csv`、`ith_scores.csv`、`habitat_basic_features.csv`、ML 的 AUC / 阈值 |
| 关键风险 | **群体级聚类对受试者顺序与随机种子敏感**，基线必须显式锁定遍历顺序与 seed |
| 容差 | 标签图要求逐体素一致；浮点特征给出显式容差并记录 |
| 环境 | `py310` conda 环境，同时记录依赖版本指纹 |

---

## 14. 风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| 数值漂移 | 高 | 阶段 0 先固化基线；每阶段对齐验证 |
| 非编程用户体验退步 | 高 | 每次改动跑第 11.2 节的无感验收 |
| 过度抽象 | 中 | 判据：**无法用生境分析的领域语言解释其存在理由的抽象，一律不要** |
| 惰性 `Subject` 与多进程交互 | 中 | 独立设计序列化与缓存边界，先在 `SerialBackend` 验证 |
| 旧 YAML 长尾（74 个模板 + 隐式默认值） | 中 | 用 `tests/test_all_configs.py` 逐个覆盖翻译层 |
| 工程量失控 | 中 | 垂直切片优先；阶段 2 不通过就停下重新评估 |

---

## 15. 开放问题（已决策）

| # | 问题 | 决策 |
|---|---|---|
| 1 | **License**：非商业条款与 G1「嵌入生态」冲突 | 已迁移至 **Apache-2.0**（含 `NOTICE`、`CITATION.cff`、上游署名）。引用从法律强制转为社区规范，由第 9 节的自动方法学描述做工程兜底 |
| 2 | **`HabitatModel` 的分发形态** | v1.0 只做**自描述的本地文件**（`.habitatmodel`），保证脱离训练输出目录可用；模型注册中心推迟到 v1.1 视社区需求评估 |
| 3 | **v0.1.x 的维护期限** | v1.0 稳定后停止维护，仅保留 tag 供复现历史结果 |
| 4 | **报告规范的对外形态** | 中间路线：先随 HABIT 发布建议清单（`manifest.checklist()`），若被采用再推动正式共识 |

执行计划与重构后的完整使用说明见 [`07_v1_refactor_plan_and_usage.md`](./07_v1_refactor_plan_and_usage.md)。
