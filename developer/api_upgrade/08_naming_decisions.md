# HABIT v1.0 命名定案（权威）

> 本文档是 v1.0 API 命名的**唯一权威依据**。它收敛了命名审查（见子代理报告）与既有 v0.1.x 代码惯例之间的冲突。云端重构与所有后续实现以此为准；`06`/`07` 文档与 `prototype/` 中任何与之冲突的命名，以本文为准。
>
> 原则：**沿用优秀开源库（MONAI / TorchIO / sklearn / PyRadiomics / lifelines / joblib / concurrent.futures）的既有词汇与契约，不发明同义词；医学领域已有强含义的词不挪作它用。**

---

## 1. 五个领域协议

| 定案名 | 说明 |
|---|---|
| `VoxelFeatureExtractor` | 保持。与 PyRadiomics `RadiomicsFeatureExtractor` 同构。 |
| `Supervoxelizer` | 保持。`-izer` 是 sklearn 构词法（`Binarizer`/`KBinsDiscretizer`）。 |
| `HabitatModelFitter` | **改**（原 `HabitatModelEstimator`）。`fit()` 返回新的 `HabitatModel` 而非 `self`，违反 sklearn 的 Estimator 契约；采 lifelines/statsmodels 的 `*Fitter`。`*Estimator` 这个名字保留给 `habit.compat.sklearn` 里真正返回 `self` 的适配器。 |
| `HabitatAssigner` | **改**（原 `HabitatMapper`）。`map` 已被 `Cohort.map(op)` 与 `ExecutionBackend.map(op, items)`（`Pool.map` 含义）占用，第三种含义会冲突；`Mapper` 还易被读成 ORM mapper。动词用 `assign`。 |
| `HabitatFeatureExtractor` | 保持。与 `VoxelFeatureExtractor` 平行。 |

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
- `slic` 归 `supervoxelizer`（个体内），`kmeans`/`gmm` 归 `habitat_model_fitter`（群体级）——**纠正 v0.1 把二者塞进同一 `clustering` 注册的结构性错误**。
- 构造/注册/自省 API：`<Registry>.create(name, **params)`、`@<Registry>.register("name")`、顶层 `list_plugins(domain)` / `get_plugin_info(name, domain)` / `get_param_schema(name, domain)` / `load_plugins()`。注册表基类 `ComponentRegistry`，`domain: ClassVar[str]`。
- `HabitatSpec` 的字段名与 domain 逐字一致，避免第四套词汇。

---

## 5. 模块 / 包名

| 定案 | 原 | 理由 |
|---|---|---|
| `habit.contracts` | 保持（不改 `habit.data`） | 已在所有文档/原型中使用，且"数据契约"准确；顶层 re-export 让多数人无需记子模块。 |
| `habit.kernels` | 保持（不改 `metrics`/`algorithms`） | 指纯数值计算（无 IO 无状态）。`habit.kernels.habitat_metrics` 作为复核公式的稳定路径。改名收益低于 churn。 |
| `habit.adapters` | 保持（不改 `habit.io`） | DataSource/ResultWriter 的落点；`compat.nnunet` 与之重复处应合并到 adapters。 |
| `habit.domain` | 保持（不改 `habit.components`） | 五个领域协议 + 内置实现所在层。 |
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
