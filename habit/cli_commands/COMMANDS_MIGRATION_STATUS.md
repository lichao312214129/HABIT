# Commands 模块迁移状态（V1）

## 标准模式

每个 `cmd_*.py` 走同一条标准链路：

1. `ConfigClass.from_file(config_path)` 加载并校验 YAML（Pydantic）。
2. 实例化对应的 **域 configurator**（来自
   `habit.core.common.configurators`），通过其 `create_*` 工厂获得装配好的服务对象。
3. 统一的错误处理与日志（基类 `BaseConfigurator` 注入 logger / output_dir）。

V0 时期使用的单一 `ServiceConfigurator` 类已在 V1 中删除，按业务域拆为：

- `HabitatConfigurator` —— habitat / 特征抽取 / radiomics / test-retest。
- `MLConfigurator` —— Holdout、KFold、ModelComparison、评估、报告、可视化。
- `PreprocessingConfigurator` —— BatchProcessor。

## 已落地命令

| Command | 配置类 | 装配入口 |
|---------|-------|---------|
| `cmd_habitat.py` | `HabitatAnalysisConfig` | `HabitatConfigurator.create_habitat_analysis()` |
| `cmd_preprocess.py` | `PreprocessingConfig` | `PreprocessingConfigurator.create_batch_processor()` |
| `cmd_extract_features.py` | `FeatureExtractionConfig` | `HabitatConfigurator.create_feature_extractor()` |
| `cmd_compare.py` | `ModelComparisonConfig` | `MLConfigurator.create_model_comparison()` |
| `cmd_ml.py` (`run_ml`) | `MLConfig` | `MLConfigurator.create_ml_workflow()` |
| `cmd_ml.py` (`run_kfold`) | `MLConfig` | `MLConfigurator.create_kfold_workflow()` |
| `cmd_radiomics.py` | `RadiomicsConfig` | `HabitatConfigurator.create_radiomics_extractor()` |
| `cmd_test_retest.py` | `TestRetestConfig` | `HabitatConfigurator.create_test_retest_analyzer()` |
| `cmd_icc.py` | `ICCConfig` | 函数式 `run_icc_analysis_from_config()`（V1 已 Pydantic 化） |

## 不需要 configurator 的命令

| Command | 说明 |
|---------|------|
| `cmd_dicom_info.py` | 无配置文件，仅命令行参数 |
| `cmd_merge_csv.py`  | 无配置文件，仅命令行参数 |
| `habit dice`        | 无配置文件；在 `habit/cli.py` 中直接调用 `habit.utils.dice_calculator` |

## 添加新命令的步骤

1. 在合适的 `config_schemas.py` 中继承 `BaseConfig` 添加新 Pydantic 配置类。
2. 在对应域的 `configurators/<domain>.py` 加 `create_<service>()` 工厂；
   重要：业务 import 写在工厂内部以保持 `common` 的延迟 import 面。
3. 在 `habit/cli_commands/commands/` 新增 `cmd_<name>.py`，复用既有
   command 的链路（加载 → 装配 → run）。
4. 在 `habit/cli.py` 注册子命令。
