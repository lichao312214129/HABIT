# HABIT v1.0 命名定案（权威）

> 本文档是 v1.0 API 命名的**唯一权威依据**。它收敛了命名审查（见子代理报告）与既有 v0.1.x 代码惯例之间的冲突。云端重构与所有后续实现以此为准；`06`/`07` 文档与 `prototype/` 中任何与之冲突的命名，以本文为准。
>
> 原则：**沿用优秀开源库（MONAI / TorchIO / sklearn / PyRadiomics / lifelines / joblib / concurrent.futures）的既有词汇与契约，不发明同义词；医学领域已有强含义的词不挪作它用。**

---

## 1. 九个领域协议

| 定案名 | 说明 |
|---|---|
| `VoxelFeatureExtractor` | 保持。与 PyRadiomics `RadiomicsFeatureExtractor` 同构。 |
| `SubjectFeaturePreprocessor` | **新增**（见 1c）。无状态、逐个体，体素/超体素两种粒度通用。 |
| `Supervoxelizer` | 保持。`-izer` 是 sklearn 构词法（`Binarizer`/`KBinsDiscretizer`）。**语义收窄**：只负责长出划分，不再决定如何描述它。 |
| `SupervoxelFeatureExtractor` | **新增**（见 1a）。与 `VoxelFeatureExtractor` / `HabitatFeatureExtractor` 三者平行，构词一致。 |
| `CohortFeaturePreprocessor` | **新增**（见 1c）。有状态、在训练队列上 fit，状态随 `HabitatModel` 流通。 |
| `HabitatModelFitter` | **改**（原 `HabitatModelEstimator`）。`fit()` 返回新的 `HabitatModel` 而非 `self`，违反 sklearn 的 Estimator 契约；采 lifelines/statsmodels 的 `*Fitter`。`*Estimator` 这个名字保留给 `habit.compat.sklearn` 里真正返回 `self` 的适配器。 |
| `HabitatAssigner` | **改**（原 `HabitatMapper`）。`map` 已被 `Cohort.map(op)` 与 `ExecutionBackend.map(op, items)`（`Pool.map` 含义）占用，第三种含义会冲突；`Mapper` 还易被读成 ORM mapper。动词用 `assign`。 |
| `HabitatFeatureExtractor` | 保持。与 `VoxelFeatureExtractor` 平行。 |
| `Combiner` | **新增**（见 1d）。多模态/多源特征块的列向组合，体素/超体素/生境三层共用。 |

### 1a. 为什么补第六个协议（推翻"恰好五个"）

初版把超体素特征提取并入 `Supervoxelizer`，依据是第 2 节对 `Supervoxelization` 的定性——"划分 + 特征"的复合产物。**该合并已推翻。**

- **两个正交轴被压成一个名字。** v0.1 用两个独立配置块表达：`habitat_segmentation.supervoxel`（算法）与 `feature_construction.supervoxel_level`（特征）。合并后 spec 只有一个 name 位置，`LegacyConfigAdapter` 只能把第二个轴降级成 warning 里的字符串。
- **`supervoxel_radiomics` 在签名上不可实现。** 它需要原始影像，而 `Supervoxelizer.__call__(field)` 只有 `VoxelFeatureField`。

签名 `__call__(Subject, Supervoxelization) -> Supervoxelization`，与 `HabitatFeatureExtractor.__call__(Subject, HabitatMap)` 同构。返回 `Supervoxelization` 而非 `FeatureTable`：下游 fitter / assigner 消费的是分区，返回裸表会逼它们多认一种输入类型。

第 2 节对 `Supervoxelization` 的定性**保持不变**——它仍是"划分 + 特征"的复合产物，只是"谁来填 features"现在有了明确答案：默认由 supervoxelizer 附上体素特征均值（零成本，且是 v0.1 默认），需要别的描述时由本协议替换。

### 1c. 为什么补第七、第八个协议：特征预处理

初版把「特征提取后、聚类前的数值处理」塞进了 `table_preprocessor` 域，`HabitatSpec` 里叫 `subject_table_preprocessors` / `group_table_preprocessors`。**该归属已推翻**，两个理由：

- **`table` 这个词指错了东西。** `TablePreprocessor` 收 `FeatureTable`——一行一个受试者、带 ID 与结局列的建模表。而这里的矩阵一行是一个体素或一个超体素，没有 ID 列也没有结局列。复用同一个协议只能靠绕过 ID 列检查，那已经说明类型不对。
- **有字段但没有消费点。** `legacy.py` 会翻译、`HabitatSpec` 会序列化，而 `SubjectPipeline` 里没有任何地方读它。这比"不支持"更危险：分析照跑、溯源照记，数字来自一条从未归一化的流水线。

**划分依据是状态是否跨受试者，不是数据粒度。** v0.1 的两个块名（`preprocessing_for_subject_level` / `preprocessing_for_group_level`）把这两件事混在了一起——个体级块永远作用于体素特征，群体级块在 `two_step` 下作用于超体素特征、在 `direct_pooling` 下又变回体素特征。按状态归属重新划分后：

| 定案名 | 状态 | 科学目的 | 可用粒度 |
|---|---|---|---|
| `SubjectFeaturePreprocessor` | 无状态 | 消除个体间差异（扫描仪、序列、强度尺度） | 体素**与**超体素 |
| `CohortFeaturePreprocessor` | 有状态 | 让不同个体的单元落进同一个可比空间 | 体素**与**超体素 |

前者必须无状态是科学要求而非实现选择：用个体自己的分布才能抹掉个体差异，用了训练队列的统计量抹掉的就不是个体差异了。因此它在训练与预测时是同一个计算，结构上不可能泄漏；后者是生境定义中**唯一**的泄漏敏感点。

`HabitatSpec` 里因此有**三条链**却只有**两个协议**：

| spec 字段 | 协议 | 对应 v0.1 |
|---|---|---|
| `voxel_feature_preprocessors` | `SubjectFeaturePreprocessor` | `preprocessing_for_subject_level` |
| `supervoxel_feature_preprocessors` | `SubjectFeaturePreprocessor` | **无对应**（v0.1 表达不出来） |
| `cohort_feature_preprocessors` | `CohortFeaturePreprocessor` | `preprocessing_for_group_level` |

内置实现类名是 `SubjectPreprocessingChain` / `CohortPreprocessingChain`，与协议名分开——沿用 `Supervoxelizer`（协议）对 `SlicSupervoxelizer`（实现）那条惯例。叫 "Chain" 而非另一个 `...Preprocessor`，是因为它本身不是一个可插拔方法，而是若干方法的**有序组合**；真正可插拔的是方法，注册在新域 `feature_preprocessing_method` 下。

#### 1c-i. `global_normalize` → `across_features`

v0.1 每个预处理方法都有 `global_normalize` 参数。这个名字读起来像"使用全局（队列级）统计量"，但它**从来不是这个意思**——它选择的是统计量在**特征列之间**汇总（一个 min/max 服务所有列）还是每列各算一份。统计量取自哪些行，由持有该方法的链决定，与这个参数无关。

在旧名旁边放上新的链名（`subject` / `cohort`）会让误读变成必然，所以改名 `across_features`。`legacy.py` 自动转换，取值原样带过。

这个区分对多模态特征是科学性的而非装饰性的：跨列汇总保留模态之间的相对强度尺度，逐列缩放会把它抹平。

`table_preprocessor` 域的同名参数**一并改成 `across_features`**。同一个 v1 版本里同一个概念不能有两个名字，否则读者会以为两者行为不同。该域没有 v0.1 翻译路径（v0.1 无此插件域），所以改名不影响任何旧配置。

#### 1c-iii. 数值内核上提到 L0，两个预处理域共用

`table_preprocessor` 与 `feature_preprocessing_method` 的八个方法数值实现原本各写了一份（再加 v0.1 那份，共三份）。已抽到 **L0 `habit.kernels.feature_transforms`**，两个 L3 域都向下依赖它。

落在 L0 而不是某个 L3 子包，是因为这些函数确实符合 L0 定义——只吃 `DataFrame`、不认识 `Subject` / `Spec` / 文件系统，不 import 任何 `habit` 模块（架构契约测试对 `habit.kernels` 的要求正是这条）。跨子包互相 import 会让"谁依赖谁"变得随机。

内核按 `fit_*` / `apply_*` 切开，"状态从哪来"完全交给调用方：无状态链对同一个矩阵 fit 又 apply，有状态链 fit 一次到处 apply。方法本身不需要知道自己服务哪种链。

**内核状态必须是纯可 JSON 序列化的**，这是硬约束而非整洁癖：队列链的状态要随 `HabitatModel` 走，而 `.habitatmodel` 刻意不是裸 pickle（见第 2 节）。因此按列统计量存成 `{列名: float}` 而不是 `Series`，`binning` 存 bin 边界而不是 fit 好的 `KBinsDiscretizer`——后者会让"带预处理的生境定义"根本存不了盘，而那恰恰是最值得分发的模型。`apply_binning` 用 `np.searchsorted(edges[1:-1], x, side="right")` 再 clip 复现 sklearn 的 ordinal 行为，对 `uniform`/`quantile`/`kmeans` 三种策略及越界值都有逐位一致的测试。

#### 1c-ii. `impute`：非有限值处理从硬编码变成一个方法

v0.1 在链的开头硬编码了一段 NaN/Inf 清理（`_prepare_feature_block` + `handle_extreme_values`）。底层 helper 本来就带 `strategy` 参数，但调用方写死了 `"mean_replacement"`，配置层永远碰不到。

v1 把它注册成普通方法 `impute`（`strategy: mean | median | zero`）。它的 fit/transform 语义天然吻合两条链：个体链每次从自己重算，队列链学一次然后复用。两条链在**未显式配置时自动前置一个默认实例，并写进 spec**——既保住 v0.1 数值兼容，又让这一步可配置、可审计（自动加的步骤也必须出现在溯源里，不能有"记录之外的计算"）。

**一处刻意的数值分歧**：v0.1 用训练均值填 NaN，紧接着用**当前块**的列均值替换 Inf。在有状态链里这等于让测试集统计量参与了自己的变换。v1 统一取训练统计量。仅当数据真的含非有限值时两者才有差异（原始强度特征通常没有，radiomics 偶有）。

### 1d. 为什么补第九个领域协议：Combiner 与特征树

HABIT 的核心科学优势是天然多模态（T1/T2/CT/PET 任意组合）。这要求：不管是体素、超体素还是生境层，**单模态提取方式**与**多模态组合方式**都应是多样、可扩展、可替换的两件正交的事。v1.0 初版不满足：`concat` 只是体素域的一个普通提取器，超体素层只有 `mean_voxel_features` 与 `supervoxel_radiomics`，组合逻辑写死在个别组件里，第三方无法替换。

**决策：组合逻辑从提取器里拆出，独立成协议 `Combiner`。** 签名 `__call__(blocks: Sequence[DataFrame], *, context: Mapping) -> DataFrame`——只吃列对齐的特征块、不认识 `Subject` / `Spec` / 文件系统，因此同一个组合器三层复用、纯 pandas 可测。这与 1c 把预处理拆成独立协议的理由同构：可插拔的粒度与提取器协议不重合时，就给可插拔物自己的域。

配套定案：

- **节点抽象**：每个提取阶段统一为递归 `Spec` 树。叶子三种形态——形态 0（几何，无模态，如 `volume`）、形态 1（单模态，`raw("t1")` / `mean("t1")`）、形态 2（原子多模态，一次调用必须同时看多个模态，如深度学习嵌入；罕见逃逸口，不鼓励）。组合器节点的 children 存在 `params["children"]`，**不给 `Spec` 加新字段**，指纹/序列化/校验零改动。树求值包装器（`VoxelFeatureTree` 等三个）实现各层既有协议，管线对叶子与树透明。
- **单模态参数 `modality`**：与既有 `modalities`（仅 `raw` 的多模态堆叠）并列、互斥；超体素/生境层的同类提取器（`supervoxel_radiomics`、`each_habitat`、`traditional`）同样补 `modality` 单模态形态。来源标签参数 `as_`（尾下划线避开 Python 关键字，pydantic 惯例）覆盖列名来源；统计输入选择 `source: "working" | "original"`（预处理后 / 预处理前体素信号）。
- **统计提取器 `mean` / `std` / `percentile`** 注册在 `supervoxel_feature_extractor` 域而**不是**组合器：它们是单模态形态 1 提取器，把同一超体素内某模态的体素信号聚成一个标量（行轴聚合、改变粒度），不满足"列向合并等粒度块"的组合器契约。管线通过 `bind_fields(working, original)` 钩子把体素场绑给它们，不改协议签名。命名沿用 pandas `groupby` 聚合词汇（`mean`/`std`/`percentile`），不造同义词；`q` 沿用 numpy/pandas 分位数参数名。
- **内置组合器名**：`concat` / `kinetic` / `expression` 沿用 v0.1 词汇（与体素域遗留叶同名是刻意的——同一表达式在旧配置走遗留叶、在新树走组合器，语义一致）；新增 `weighted_concat` / `average` / `ratio` / `difference`（`ratio`/`difference` 恰两个子节点，取名自运算本身）。
- **列命名规则**：单列节点列名 = 来源标签（`as_` > `modality` > 组件名，如 `raw("T1")` → `T1`）；多列节点 `{feature}-{source}`（如 `local_entropy-T2`）；`percentile` 家族前缀 `p{q}`（如 `p90-T2`）。`as_` **只允许单列输出节点**，多列节点带 `as_` 直接报错——允许部分列改名会让剩余列名不可预测，静默歧义比报错更糟。
- **表达式 DSL 是树的严格投影**：`habit.spec.parse_feature_expression`，引号模态、显式 `key=value`、嵌套调用、children 中的引号串自动成为 `raw` 叶。v0.1 宽松语法只留在 legacy adapter：表达式**含引号**才路由到新解析器，旧配置逐字节一致、新配置得树；歧义输入（裸标识符等）硬报错不猜。YAML 双写法（结构化 mapping 与表达式字符串）指纹逐位一致，由 `coerce_spec` 路由。

### 1b. 调用约定：单一 `__call__`，**删除所有动词别名**

- **删除** `extract = __call__` / `build = __call__` / `map = __call__` 这类类体别名。原因：类体别名绑定的是定义时刻的函数对象，子类覆写 `__call__` 后别名仍指向父类实现，造成**静默分叉**；且 `@runtime_checkable` 协议若要求两个名字，会抬高第三方实现门槛。MONAI/TorchIO/PyRadiomics/sklearn 都不用双公开名。
- 可读性来自**调用点变量名**（`voxel_features(subject)`），不来自第二个方法名。
- 删除 `HabitatFeatureExtractor.name` 属性——注册名已在 `spec.name`，两处只留一处。

---

## 2. 数据契约（`habit.contracts`）

| 定案名 | 说明 |
|---|---|
| `Geometry` / `ImageVolume` / `MaskVolume` | 保持。 |
| `ImageRef` | 保持（不改 `ImageProxy`）。**但** `ImageVolume`/`MaskVolume` 应在结构上满足 `ImageRef`（`load()` 返回自身数组），只留一族类型，不搞 eager/lazy 双轨。 |
| `Subject` / `Cohort` | 保持（`= tio.Subject`，全表最佳命名）。 |
| `VoxelFeatureField` | 保持（不改 `VoxelFeatures`）。它是带 `voxel_index` 与 `geometry` 的稀疏行表，"Field"强调可渲染回图像空间；改名收益不足以抵消 churn。 |
| `Supervoxelization` | 保持（不改 `SupervoxelMap`）。它不只是标签图，还含每个超体素的 `features` DataFrame，是"划分+特征"的复合产物；与 `HabitatMap` 本就不是同一类型。 |
| `HabitatMap` / `HabitatModel` / `FeatureTable` / `Provenance` / `CohortFingerprint` | 保持。 |

### 2b. 契约方法/字段

| 定案 | 原 | 理由 |
|---|---|---|
| `HabitatModel.summary()` | `describe()` | pandas `.describe()` 返回统计表，是最强先验；返回散文用 statsmodels 的 `summary()`。 |
| `HabitatModel.assigner()` | `mapper()` | 随协议改名同步。 |
| `Cohort.summarize() -> CohortFingerprint` | `fingerprint()` | 返回富对象，与 `Spec.fingerprint()`（返回哈希字符串）分家；返回类型仍叫 `CohortFingerprint`（nnU-Net 用词）。 |
| `FeatureTable.outcome_column` | `label_column` | 原字段 docstring 自己写的就是 "Outcome column"，代码与文档打架；outcome 是医学终点标准词。 |
| `FeatureTable.feature_matrix()` | `features()` | 避免与"跑特征提取"混淆；返回的是矩阵式 frame。 |
| `VoxelFeatureField.feature_frame()` / `.with_feature_frame()`<br>`Supervoxelization.feature_frame()` / `.with_feature_frame()` | 新增 | 两个契约的**对称方法对**，是"存储不统一、接口统一"的落点。取名 `feature_frame` 而非 `to_frame`，因为后者已存在且会插入 z/y/x 坐标列（供人看）；这一对只给算法用，必须是裸特征矩阵。 |
| `HabitatModel.with_cohort_preprocessing()` | 新增 | `with_*` 表示返回新实例（契约是 frozen）。它同时**重算 `model_id`**：质心来自不同特征空间的两个模型是两个不同的生境定义，不能共用标识。 |

---

## 3. 执行 / 基础设施（`habit.execution` 等）

| 定案名 | 原 | 理由 |
|---|---|---|
| `SubjectOperator` / `CohortOperator` | `SubjectLevelOp` / `CohortLevelOp` | 设计正文已写"算子/operator"；`Op` 是缩写且在医学库易误读为"手术"。 |
| `SubjectResult`（`.result()`） | `Outcome`（`.unwrap()`） | **最关键改名**。outcome 在医学=被预测的临床结局；`concurrent.futures.Future.result()` 是标准库锚点。并补上 `@dataclass(frozen=True)`（原来缺）。 |
| `ExecutionBackend` / `ProcessPoolBackend` / `CheckpointStore` | 保持 | 与 joblib/concurrent.futures 对齐。 |
| `SerialBackend` | 保持（不改 `SequentialBackend`） | 低优先级；`SerialBackend` 已清晰，改名收益低。 |
| `DataSource.load()` | `cohort()` | 无参方法用动词（nibabel/MONAI reader 惯例），名词读着像属性。 |
| `ResultWriter` | `ArtifactSink` | "sink"是数据流黑话；"artifact"在放射学=伪影，是真撞名。方法都是 `write_*`，故叫 Writer（对齐 `monai.data.ImageWriter`）。 |
| `Seedable` | `SeedControl` | 形容词协议（`Iterable`/`Hashable`/MONAI `Randomizable`）。不叫 `Randomizable` 是因 MONAI 签名不同（`set_random_state(seed=None, state=None)->self` + `randomize()`），避免重蹈 Estimator 覆辙。 |
| `set_random_state(seed: int) -> None` | 保持 | 与 MONAI 同名。 |
| `RunPolicy` / `Spec` / `HabitatSpec` | 保持 | 切分干净；`Spec` 在 `habit.spec` 模块内不泛化。 |

---

## 4. Registry 与插件域（**最需收敛**）

**规则：`domain == snake_case(协议类名)`，单数，agent-noun。** 实现了某协议的人不看文档就知道它的域。entry point 分组 = `habit.<domain>`。

| domain（定案） | 协议 | entry point 组 |
|---|---|---|
| `voxel_feature_extractor` | `VoxelFeatureExtractor` | `habit.voxel_feature_extractor` |
| `supervoxelizer` | `Supervoxelizer` | `habit.supervoxelizer` |
| `supervoxel_feature_extractor` | `SupervoxelFeatureExtractor` | `habit.supervoxel_feature_extractor` |
| `feature_preprocessing_method` | （两个预处理协议共用，见下） | `habit.feature_preprocessing_method` |
| `habitat_model_fitter` | `HabitatModelFitter` | `habit.habitat_model_fitter` |
| `habitat_assigner` | `HabitatAssigner` | `habit.habitat_assigner` |
| `habitat_feature_extractor` | `HabitatFeatureExtractor` | `habit.habitat_feature_extractor` |
| `combiner` | `Combiner` | `habit.combiner` |
| `pooling` | `PoolingMarker` | `habit.pooling` |
| `preprocessor` | 图像预处理 | `habit.preprocessor` |
| `table_preprocessor` | 特征表预处理 | `habit.table_preprocessor` |
| `classifier` | ML 模型 | `habit.classifier` |
| `feature_selector` | 特征选择 | `habit.feature_selector` |
| `metric` | 评估指标 | `habit.metric` |

`pooling` / `PoolingMarker`：有序 stage 列表里的 **subject→cohort 分水岭标记**
（内置名 ``pool``）。它不是聚类算法；executor 识别该角色后调用
``habit.pipeline.pooling.fan_in``。见 `07` §F（2026-08-09 stages 定稿）。

要点：
- **单数、协议名**，不是复数域名（推翻此前的 `habit.preprocessors` 复数方案——那是 v0.1 的偶然，不是该继承的约定；v1.0 允许破坏性重构，统一为协议名更自洽）。
- `model` → `classifier`：把 `HabitatModel`（生境定义产物）与 ML 分类器彻底分开。
- `slic` 归 `supervoxelizer`（个体内），`kmeans`/`gmm` 归 `habitat_model_fitter`（群体级）——**纠正 v0.1 把二者塞进同一 `clustering` 注册的结构性错误**。注意 `kmeans`/`gmm` 在 `supervoxelizer` 域**也各有一个同名实现**（v0.1 的 `supervoxel.algorithm: kmeans|gmm`，在特征空间对个体内体素聚类）：同名不同域是对的，因为域已经区分了个体级与群体级，强行改名反而丢掉 v0.1 的既有词汇。
- 内置组件名一律沿用 v0.1 的 YAML 写法（`mean_voxel_features`、`supervoxel_radiomics`），旧配置的 method 名可直接翻译成 spec name。
- `feature_preprocessing_method` 是**唯一一个域名不等于协议名**的例外，理由是可插拔的粒度在此处与协议不重合：两个预处理协议的实现是**链**（方法的有序组合），而第三方要插的是链里的一个**方法**。域名描述可插拔物，所以叫 `..._method`。同一个注册表同时服务两条链——一个方法不需要知道自己处理的是体素还是超体素，也不需要知道持有它的链会丢弃还是保存它的状态。
- 注意与 `table_preprocessor` 域的区别：那个域预处理**建模表**（一行一受试者，通往结局模型），这个域预处理**聚类输入**（一行一体素/超体素，通往生境定义）。两者数值实现相同而行语义不同，是两个域。
- 构造/注册/自省 API：`<Registry>.create(name, **params)`、`@<Registry>.register("name")`、顶层 `list_plugins(domain)` / `get_plugin_info(name, domain)` / `get_param_schema(name, domain)` / `load_plugins()`。注册表基类 `ComponentRegistry`，`domain: ClassVar[str]`。
- `HabitatSpec` 的字段名与 domain 逐字一致，避免第四套词汇。
- **`HabitatComponents` 属性名也与 Spec / `SubjectPipeline` 对齐**（不另造缩写）。装配袋里不得再出现 `voxel_extractor` / `supervoxel_extractor` / `voxel_chain` / `supervoxel_chain` / `cohort_chain` / `fitter` / `extractors`；对应为 `voxel_feature_extractor` / `supervoxel_feature_extractor` / `voxel_feature_preprocessor` / `supervoxel_feature_preprocessor` / `cohort_feature_preprocessor` / `habitat_model_fitter` / `habitat_features`。Spec 上预处理字段仍用复数（`*_preprocessors`，一步列表），装配后的链用单数（与 pipeline 参数一致）。工厂函数 `build_voxel_extractor` 等是树节点构造器，不是 Components 属性名。

---

## 5. 模块 / 包名

| 定案 | 原 | 理由 |
|---|---|---|
| `habit.contracts` | 保持（不改 `habit.data`） | 已在所有文档/原型中使用，且"数据契约"准确；顶层 re-export 让多数人无需记子模块。 |
| `habit.kernels` | 保持（不改 `metrics`/`algorithms`） | 指纯数值计算（无 IO 无状态）。`habit.kernels.habitat_metrics` 作为复核公式的稳定路径；`habit.kernels.feature_transforms` 是两个预处理域共用的 fit/apply 内核（见 1c-iii）。改名收益低于 churn。 |
| `habit.adapters` | 保持（不改 `habit.io`） | DataSource/ResultWriter 的落点；`compat.nnunet` 与之重复处应合并到 adapters。 |
| `habit.domain` | 保持（不改 `habit.components`） | 九个领域协议 + 内置实现所在层。子包名与 domain 对应：`supervoxel/`（划分）与 `supervoxel_features/`（描述）分列，`feature_preprocessing/`（聚类输入预处理）与 `table_preprocessing/`（建模表预处理）分列，`combiners/` 放列向组合器，都和 `habitat_features/` 构词一致。 |
| `habit.execution` / `habit.registry` | 保持 | 直白准确。 |
| `habit.spec` | 保持（不拆） | `Spec`/`RunPolicy`/YAML 同构/legacy 翻译集中于此；核心算法不 import YAML 即可满足"核心不知 YAML"——靠 import 约束而非拆包。 |
| `habit.recipes` | 保持（不改 `habit.workflows`） | 一行式配方（`recipes.two_step_habitat()`）。CLI 的 `--workflow` 是另一概念，不冲突。 |
| `habit.compat` | 保持（不改 `habit.integrations`） | `compat.sklearn`/`compat.monai`/`compat.nnunet` 第三方互操作。 |

> 说明：命名审查建议改 `habit.data`/`habit.io`/`habit.components`/`habit.workflows`/`habit.integrations`。这些在纯美学上各有道理，但都属于"可改可不改"，而 v1.0 已有大量文档与原型引用现名。**只有当现名造成真实歧义时才改**（如 `Outcome`、`ArtifactSink`、`HabitatMapper`），否则保持稳定、降低实现与审查成本。
>
> **2026-09-02 增补**：API 参考页上同一族对象一半写 `habit.X`、一半写
> `habit.domain.X`，即"真实歧义"。因此 `habit.domain` 不再作为 L3 实现层的物理
> 名字：按能力命名的 `habit.voxel_features` / `habit.supervoxel` /
> `habit.habitat_model` / … 同时是**规范用户路径和真实代码位置**。v2.0.0 删除
> `habit.domain.*` 与旧的扁平组件导入，不保留 compatibility re-export。命名表、分层门禁
> 与分阶段迁移计划见
> [`09_capability_namespaces.md`](09_capability_namespaces.md)。
>
> **2026-09-02 v2 参数契约增补**：组件的 `__init__` 是唯一参数真相来源。
> 删除所有组件 `*Params` Pydantic 配套模型与 Registry `params_model`；Registry
> 只保留名称到类的映射及 `create(name, **kwargs)`。构造器以 sklearn 风格的类型标注、
> 默认值、允许值/数值边界、运行期校验和同名实例属性表达契约；参数自省与自动文档直接
> 读取 `inspect.signature`、注解和 `Parameters` docstring。`Spec.params` 仍是传给
> 构造器的无类型 kwargs/溯源字典，不再承担 Pydantic schema。此项为 v2 破坏性迁移，
> 不保留旧 Params 或 `get_params_model()` compatibility。

---

## 6. 配方函数名

统一为 `habit.recipes.two_step_habitat()` / `one_step_habitat()` / `direct_pooling_habitat()`（动词在后、可读作 "two-step habitat"）。文档 07 中出现的 `habitat_two_step` 一律改为此。

---

## 7. 其它已定

- 顶层便捷：`habit.cohort_from_directory(...)`（自由函数，notebook 友好）与 `Cohort.from_directory(...)`（类方法）**两者都保留**，前者内部调后者。
- 结果对象：`StudyResult.habitat_model`（不是 `.model`）、`.pipeline`、`.features`、`.habitat_maps`、`.manifest`。
- 报告 API 集中在 `RunManifest`：`describe_methods(style=...)`、`checklist(standard=...)`、`software_versions()`、`random_seeds()`、`to_json()`。
- 失败策略取值：`"continue"` / `"fail_fast"`；并发度参数 `workers`（不是 `processes`）。
- 超体素个数参数 `n_supervoxels`；生境个数 `n_habitats`；随机种子统一 `set_random_state(seed)`（不在构造器里塞 `random_state`）。
- 异常基类：`HabitError`（`HABITAPIError` 为其 API 层子类）。
- **保留键 `estimator_params`**：B 类薄包装组件（政策三分见 `06` §8.4）接收厂商长尾参数的**唯一**键名，常量为 `habit.utils.estimator_utils.ESTIMATOR_PARAMS_KEY`。不造同义词（`kwargs` / `extra_params` / `vendor_params`）：`kwargs` 是 Python 语法词而非领域词，`extra`/`vendor` 都不说"这是给底层 estimator 的"。该键非空时并入 `spec.params` 进指纹；`random_state` 永远禁止出现在其中（种子只走 `set_random_state`）。

---

## 8. v1.1 契约变更：`TablePipeline` 成为 `sklearn.pipeline.Pipeline` 子类

`TablePipeline` 原本是自己写的组合器。v1.1 起它**继承** `sklearn.pipeline.Pipeline`，理由是组合语义只应该有一份实现：继承之后 `clone` / `get_params` / `set_params` / 嵌套参数寻址 / `GridSearchCV` / `cross_val_score` 全部免费获得，而自己再写一套只会与生态那套慢慢漂移，且漂移无人察觉。

下面三条是**行为变更**（签名未变而语义变了，按 `06` 的定义同属破坏性变更），故在此登记理由与新契约。

### 8.1 `.steps` 归 sklearn 语义，HABIT 组件改走 `.components`

| | v1.0 | v1.1 |
|---|---|---|
| `pipeline.steps` | `Tuple[HABIT 组件, ...]` | `List[Tuple[str, estimator]]`（sklearn 语义） |
| HABIT 变换组件 | `pipeline.steps` | **`pipeline.components`** |
| 终端模型 | `pipeline.model` | `pipeline.model`（不变） |

**为什么不覆写 `.steps` 保持旧语义**：sklearn 的 `Pipeline._iter` / `_validate_steps` / `_get_params` / `_set_params` / `_fit` 直接读**并写** `self.steps`（`_fit` 里有 `self.steps = list(self.steps)`，`_replace_estimator` 里有 `setattr(self, attr, new_items)`）。把它变成只读 property 会直接坏掉父类。所以 `.steps` 让给 sklearn，HABIT 视角新开 `.components`——它是**解包后**的组件元组，跳过 `FrameToTable` 头步与终端模型适配器。

**为什么不叫 `.habit_steps` / `.operators`**：`06` 的分层文档一直把这些对象称作"组件"（component），registry 也叫"组件注册表"。沿用已有领域词，不另造同义词。

**步名是稳定的公开字符串**：头步恒为 `"frame_to_table"`，终端恒为 `"model"`，中间步取组件的注册名（`"zscore"` / `"variance"` / `"lasso"`），重名追加 `_2`。这样 `param_grid={"model__component__C": [...]}` 是可写进文档的固定写法，而不是每条流水线都要先去发现一遍。

### 8.2 `FrameToTable` 头步：为什么流水线一定带一个

sklearn 的交叉验证驱动要对 `X` **按行切片**，而 `FeatureTable` 是 frozen dataclass、故意不可按行索引——实测 `cross_val_score(pipe, FeatureTable, y)` 直接死在 sklearn 的入参校验里。所以：**`X` 传原始 `DataFrame`（id 列 + 特征列 + outcome 列都在里面），流水线第一步按静态 schema 把它重建成 `FeatureTable`**。schema（哪些是 id 列、endpoint 是什么）是元数据不是数据，行重采样不改变它，所以它是构造参数，能原样穿过 `clone`。

头步在收到 `FeatureTable` 时**原样返回**：不走 frame 往返、不发生 dtype 提升、不重排列。这是数值要求而不是优化——float32 表经 `DataFrame` 重建可能移动后续 z-score 学到的队列统计量。

### 8.3 适配器 `classes_` 改为 endpoint 原生 dtype

HABIT 分类器的概率**帧**用 `str(label)` 作列名（这让帧可读），v1.0 的 `TableClassifierEstimator.classes_` 直接照抄了这些列名，于是同一个适配器 `predict()` 返回 `0/1` 整数、`classes_` 却说 `['0','1']`——**自相矛盾**，且 sklearn 的打分器要拿 `classes_` 去和它收到的 `y` 对齐，`cross_val_predict(..., method="predict_proba")` 这类路径会直接报错。

v1.1 起：
- `classes_` = endpoint 原生 dtype 的标签，顺序与概率帧列一致（`str()` 往返不唯一时退回列名，因为错 dtype 好过错**顺序**——顺序决定哪一列是正类）；
- 新增 `proba_columns_` 承载概率帧列名，列对齐一律走它。

受影响的门禁断言：`tests/compat/test_sklearn_compat.py::test_table_classifier_full_surface`、`::test_table_classifier_outcome_contract`（两处 `set(classifier.classes_) == {"0","1"}` → `{0, 1}`）。

---

## 9. v1.1 契约变更：`MLSpec` 的表步骤收敛为单一有序 `steps`

`MLSpec` 原本用三个固定字段表达顺序：

```
pre_preprocessing_feature_selectors  →  table_preprocessors  →  feature_selectors
```

这是**用槽位表达顺序**。它只提供两个可选位置——全部预处理之前、全部预处理之后——所以 `zscore → variance → minmax → lasso` 这种顺序根本写不出来。这正是"配置是界面不是架构"的反模式：顺序是领域事实（z-score 之后每个特征方差都是 1.0，方差筛选在那里就是空操作），却被编码成了数据结构的形状。

v1.1 起 `MLSpec.steps: Tuple[Spec, ...]` 是**唯一**的表步骤表达：**列表顺序就是执行顺序**。三个旧字段保留为弃用别名，整个 v1.x 全程可用。

### 9.1 名字跨注册表解析，不加 kind 标签

`steps` 里每一项只有 `name`。解析发生在 L3 的 `habit.pipeline.assembly.build_table_step`：先查 `TablePreprocessorRegistry`，再查 `FeatureSelectorRegistry`。

**为什么不给每项加 `kind: preprocessor|selector`**：那会成为注册表已经知道的事情的第二份事实源，两份就会漂移。两个词表当前无重名（`variance` / `variance_filter`、`correlation` / `correlation_filter`，见 §10），因此名字足以唯一确定组件；万一将来重名，`build_table_step` **报错而不是猜**，并由 `tests/domain/test_table_pipeline.py::test_the_two_step_registries_share_no_names` 提前拦住。

**为什么解析不在 `habit.spec`**：`habit.spec` 在栈底，不得向上依赖 `habit.domain`（`tests/test_architecture_contracts.py` 强制）。所以 spec 层只记录顺序，保持 registry-free。

### 9.2 序列化形状不对称——这一条是硬约束

| 声明方式 | `to_dict()` 输出的键 |
|---|---|
| 三个旧字段（或**什么都不声明**） | `pre_preprocessing_feature_selectors` / `table_preprocessors` / `feature_selectors`，**没有** `steps` |
| `steps=` | `steps`，**没有**三个旧键 |

**为什么不无条件同时写两者**：`MLSpec.to_dict()` 的哈希就是 `provenance.spec_fingerprint`，`tests/golden/baseline/ml_kfold.json` 逐值记录了它以及展平后的 `spec_payload.*` 键集。无条件新增一个 `steps` 键会移动**每一个已发表分析**的指纹。所以"记录偏离默认值的东西"这条既有规则（`estimator_params`、`keep_at_least_one` 同一套）在这里也适用：什么都没声明的 spec 也走旧形状，因为绝大多数已发表 spec 属于这一类。

**代价，如实记录**：同一条流水线用两种写法声明会得到**两个不同的指纹**。这是有意的——指纹标识的是**文档**，两个文档确实不同；规避这一点必须选一种形状做归一化，而选 `steps` 就会移动全部历史指纹。

**旧字段 + `steps` 同时出现**：内容一致时接受（`dataclasses.replace` 会把所有字段重新传一遍，recipes 折叠 seed 覆盖时就走这条路），内容不一致时 `HABITAPIError`。不做优先级消歧：那等于运行一条文档同时声称自己没在运行的流水线。

### 9.3 v0 YAML 翻译仍然输出旧三桶键

`habit/spec/legacy.py` 的 v0→v1 翻译**继续**输出三个弃用键，`MLSpec` 再把它们折进 `steps`。于是 `before_z_score` 现在决定的确实是 `steps` 里的**位置**，但被翻译出的**文档本身**逐字节不变，指纹也不变。若让翻译直接输出 `steps`，全部历史 v0 配置的指纹都会移动——这正是 golden baseline 应当拦住的回归。

反向翻译（`habit/recipes/yaml_runner.py:_v0_selection_methods_from_spec`）两种布局都读：`before_z_score` = "该选择器之前没有任何预处理器"。v0 表达不了的交错顺序会塌回同一个布尔值，但那个 v0 payload 只用于**加载特征表**，有科学意义的顺序由 `MLSpec.steps` 直接带进 `build_table_pipeline`。

---

## 10. v1.1：`variance` / `variance_filter`、`correlation` / `correlation_filter` 收敛为单一实现

同一算法在两个注册表里注册两次、名字不同，动机就是 §9 的位置问题：作为 preprocessor 能放进预处理链中间，作为 selector 只能在两端。§9 让位置变成纯粹的位置之后，重复注册的动机消失。

**收敛方式是别名 + 参数名翻译 + 默认值保留，不是合并**，因为两者在退化情形下行为**本来就不同**：

| | `variance_filter`（preprocessor） | `variance`（selector） |
|---|---|---|
| 参数名 | `variance_threshold` | `threshold` / `top_k` / `top_percent` |
| 无列存活时 | 保留方差最大的一列（v0.1 规则） | 不保留任何列 |

这条差异是**真实的**：预处理链清空特征块会让后面每一步死在无关错误上（`check_array`: "at least one array or dtype is required"），而选择器"没有特征通过这个阈值"是合法结论。所以它成为显式参数 `keep_at_least_one`，两个名字各自默认成它一直以来的行为（filter=`True`，selector=`False`），并且**仅在偏离该默认值时**写入 `spec.params`——同 §9.2 的理由，否则每一个 `variance` 指纹都会移动。

`correlation` / `correlation_filter` 只差参数拼写（`threshold`/`method` vs `corr_threshold`/`corr_method`）与默认值（0.8 vs 0.95），没有退化分歧：贪心左到右扫描永远保留第一列。

四个名字全部继续可用。唯一实现在 `habit/kernels/feature_transforms.py`（`select_variance_columns` / `select_correlation_columns`）——同一公式两处需要时下沉到 L0，这是既有反模式条款的正解。

---

## 11. v1.1：超参搜索 recipe 与 nested CV 的命名与契约

新增公开符号：`recipes.search_hyperparameters` / `recipes.SearchResult`，以及 `cross_validate` 的 `inner_cv` / `param_grid` / `strategy` / `n_iter` / `objective` 关键字参数与 `CVResult.fold_best_params` 字段。全部为**纯增量**：既有 `cross_validate(table, spec, n_splits=..., seed=...)` 调用逐值不变（`tests/recipes/test_hyperparameter_search.py::test_plain_cross_validation_reports_no_tuning`）。

### 11.1 为什么最优参数写回 `MLSpec` 而不是返回 `best_estimator_`

调好参的模型是一个**定义**，不只是一个拟合好的对象。写回 spec 之后它照旧有指纹、能 `to_dict()` 回 YAML、能被别人重跑；只返回 `best_estimator_` 会让溯源链**断在选参那一刻**——论文里"我们网格搜索了 C"这句话再也对不上任何可复现的记录。

写回时保留原 spec 的**字段布局**（三桶 vs 单一 `steps`），理由同 §9.2：`MLSpec.to_dict()` 的两种形状不同，若调参顺手把布局迁移了，同一分析在调参前后指纹会因与调参无关的原因而变化。这一条不能用 `dataclasses.replace` 实现——它会同时重填派生的 `steps` 与三桶，只改一边正是 `MLSpec` 要拒绝的情形，所以有 `_spec_with_layout`。

### 11.2 网格键为什么是 `"<step>__component__<param>"`

沿用 sklearn 自己的嵌套寻址语法，不另造 HABIT 方言：`TablePipeline` 现在就是 `sklearn.pipeline.Pipeline`（§8），`component` 段就是适配器里包着的 HABIT 组件。终端模型的步名恒为 `"model"`，其余步名 = 组件注册名（重名时加 `_2` 后缀）。

**键在搜索开始前就解析**，无法写回 spec 的键直接报错。理由：跑完一轮长搜索才发现结果无处可记，比不搜更糟。

### 11.3 `objective` 而不是 `scoring`

参数收的是**已注册的 HABIT 度量名**（`auc` / `f1_score` / ...），不是 sklearn 的 scorer 字符串——沿用领域已有词汇，不造同义词。方向由度量自己的 `greater_is_better` 决定，调用方不需要手动取负；sklearn 恒定最大化，符号翻转封在 `_objective_scorer` 里，`SearchResult.best_score` 与 `trials[*]["mean_score"]` 一律是度量本身的数。

打分走 `TablePipeline.evaluate`，所以搜索优化的就是最终报告打印的那个量，四种 endpoint 家族的分派复用既有实现。目标度量的**解析**目前只查分类度量注册表，因为 `MLSpec.classifier` 经 `ClassifierRegistry` 装配，L4 recipe 能表达的终端模型只有分类器；等 `MLSpec` 长出回归/生存终端时，`_objective_metric` 是唯一要跟着改的地方。

### 11.4 折的来源是 HABIT 而不是 sklearn 的 splitter

搜索的 cv 传的是 `habit.evaluation.split.kfold_indices` 生成的显式 `(train, val)` 索引对，不是 `cv=5`。这样同 `n_splits` + 同 seed 下，搜索与 `cross_validate` 的划分**逐行一致**，划分逻辑只有一个源头。

### 11.5 `inner_cv` 与 `param_grid` 必须成对出现

只给一个都是静默错误，所以两个都报错：给了网格没给 `inner_cv`，调参只能发生在外层验证行上（泄漏，且表现为一个偏好的分数）；给了 `inner_cv` 没给网格，则是对空集搜索。嵌套运行的 manifest 记录的是**未调参的 spec**——外层每折各自选了不同参数，把其中之一记成"那个 spec"会把报告的指标错误归因给一个只见过部分数据的定义。

### 11.6 不引入 Optuna

`strategy` 只有 `"grid"` / `"random"`，未知值直接报错。贝叶斯/进化后端是新的硬依赖，按依赖政策它得走可选依赖 + `OptionalDependencyError`；本次不留半成品扩展口。

---

## 附：prototype 与文档 07 已收敛的矛盾

以下曾在 prototype 与 07 文档间互相矛盾，**以本定案为准**：

| 概念 | 定案 |
|---|---|
| 配方函数名 | `recipes.two_step_habitat()` |
| 数据源入口 | `DataSource.load()` |
| 构建队列 | `habit.cohort_from_directory()` 与 `Cohort.from_directory()` 并存 |
| Schema 查询 | `get_param_schema(name, domain)`（沿用 v0.1 函数，参数顺序 name 在前） |
| 组件实例化 | `<Registry>.create(name, **params)`（删除 `get_component`） |
| 受试者元数据 | `Subject.metadata` |
| 结果里的模型 | `result.habitat_model` |
| 失败策略 | `"continue"` / `"fail_fast"` |
| 并发度 | `RunPolicy.workers` |
| 方法学文本 | `manifest.describe_methods()` |
| 超体素个数 | `n_supervoxels` |
| 生境个数 | `n_habitats` |
| 随机种子 | `set_random_state(seed)` |
| 异常基类 | `HabitError` |
| 表导出 | `result.features.frame.to_csv()`（`FeatureTable` 无 `to_csv`，用其 `.frame`） |
