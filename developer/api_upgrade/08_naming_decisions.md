# HABIT v1.0 命名定案（权威）

> 本文档是 v1.0 API 命名的**唯一权威依据**。它收敛了命名审查（见子代理报告）与既有 v0.1.x 代码惯例之间的冲突。云端重构与所有后续实现以此为准；`06`/`07` 文档与 `prototype/` 中任何与之冲突的命名，以本文为准。
>
> 原则：**沿用优秀开源库（MONAI / TorchIO / sklearn / PyRadiomics / lifelines / joblib / concurrent.futures）的既有词汇与契约，不发明同义词；医学领域已有强含义的词不挪作它用。**

---

## 1. 八个领域协议

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
| `preprocessor` | 图像预处理 | `habit.preprocessor` |
| `table_preprocessor` | 特征表预处理 | `habit.table_preprocessor` |
| `classifier` | ML 模型 | `habit.classifier` |
| `feature_selector` | 特征选择 | `habit.feature_selector` |
| `metric` | 评估指标 | `habit.metric` |

要点：
- **单数、协议名**，不是复数域名（推翻此前的 `habit.preprocessors` 复数方案——那是 v0.1 的偶然，不是该继承的约定；v1.0 允许破坏性重构，统一为协议名更自洽）。
- `model` → `classifier`：把 `HabitatModel`（生境定义产物）与 ML 分类器彻底分开。
- `slic` 归 `supervoxelizer`（个体内），`kmeans`/`gmm` 归 `habitat_model_fitter`（群体级）——**纠正 v0.1 把二者塞进同一 `clustering` 注册的结构性错误**。注意 `kmeans`/`gmm` 在 `supervoxelizer` 域**也各有一个同名实现**（v0.1 的 `supervoxel.algorithm: kmeans|gmm`，在特征空间对个体内体素聚类）：同名不同域是对的，因为域已经区分了个体级与群体级，强行改名反而丢掉 v0.1 的既有词汇。
- 内置组件名一律沿用 v0.1 的 YAML 写法（`mean_voxel_features`、`supervoxel_radiomics`），旧配置的 method 名可直接翻译成 spec name。
- `feature_preprocessing_method` 是**唯一一个域名不等于协议名**的例外，理由是可插拔的粒度在此处与协议不重合：两个预处理协议的实现是**链**（方法的有序组合），而第三方要插的是链里的一个**方法**。域名描述可插拔物，所以叫 `..._method`。同一个注册表同时服务两条链——一个方法不需要知道自己处理的是体素还是超体素，也不需要知道持有它的链会丢弃还是保存它的状态。
- 注意与 `table_preprocessor` 域的区别：那个域预处理**建模表**（一行一受试者，通往结局模型），这个域预处理**聚类输入**（一行一体素/超体素，通往生境定义）。两者数值实现相同而行语义不同，是两个域。
- 构造/注册/自省 API：`<Registry>.create(name, **params)`、`@<Registry>.register("name")`、顶层 `list_plugins(domain)` / `get_plugin_info(name, domain)` / `get_param_schema(name, domain)` / `load_plugins()`。注册表基类 `ComponentRegistry`，`domain: ClassVar[str]`。
- `HabitatSpec` 的字段名与 domain 逐字一致，避免第四套词汇。

---

## 5. 模块 / 包名

| 定案 | 原 | 理由 |
|---|---|---|
| `habit.contracts` | 保持（不改 `habit.data`） | 已在所有文档/原型中使用，且"数据契约"准确；顶层 re-export 让多数人无需记子模块。 |
| `habit.kernels` | 保持（不改 `metrics`/`algorithms`） | 指纯数值计算（无 IO 无状态）。`habit.kernels.habitat_metrics` 作为复核公式的稳定路径；`habit.kernels.feature_transforms` 是两个预处理域共用的 fit/apply 内核（见 1c-iii）。改名收益低于 churn。 |
| `habit.adapters` | 保持（不改 `habit.io`） | DataSource/ResultWriter 的落点；`compat.nnunet` 与之重复处应合并到 adapters。 |
| `habit.domain` | 保持（不改 `habit.components`） | 八个领域协议 + 内置实现所在层。子包名与 domain 对应：`supervoxel/`（划分）与 `supervoxel_features/`（描述）分列，`feature_preprocessing/`（聚类输入预处理）与 `table_preprocessing/`（建模表预处理）分列，都和 `habitat_features/` 构词一致。 |
| `habit.execution` / `habit.registry` | 保持 | 直白准确。 |
| `habit.spec` | 保持（不拆） | `Spec`/`RunPolicy`/YAML 同构/legacy 翻译集中于此；核心算法不 import YAML 即可满足"核心不知 YAML"——靠 import 约束而非拆包。 |
| `habit.recipes` | 保持（不改 `habit.workflows`） | 一行式配方（`recipes.two_step_habitat()`）。CLI 的 `--workflow` 是另一概念，不冲突。 |
| `habit.compat` | 保持（不改 `habit.integrations`） | `compat.sklearn`/`compat.monai`/`compat.nnunet` 第三方互操作。 |

> 说明：命名审查建议改 `habit.data`/`habit.io`/`habit.components`/`habit.workflows`/`habit.integrations`。这些在纯美学上各有道理，但都属于"可改可不改"，而 v1.0 已有大量文档与原型引用现名。**只有当现名造成真实歧义时才改**（如 `Outcome`、`ArtifactSink`、`HabitatMapper`），否则保持稳定、降低实现与审查成本。

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
