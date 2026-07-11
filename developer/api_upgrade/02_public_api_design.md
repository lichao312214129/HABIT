# 02 — 公开 API 设计

## 1. 设计原则

1. **CLI 与 API 同源**：所有 `habit xxx` 命令最终调用 `run_*_from_config` 或等价 runner；公开 API 不引入第二套执行路径。
2. **Lazy import**：顶层 `import habit` 不得 eager 加载 `radiomics` / `torch` / `shap`；沿用 `habit.utils.lazy_exports.lazy_getattr`。
3. **稳定 vs 内部**：仅 `habit` 与 `habit.api.*` 子模块中 `__all__` 列出的符号承诺 semver；`habit.core.*` 内部路径允许在 minor 版本调整。
4. **三种路径同等文档化**：Pipeline（config runner）、Component（算法类）、Config（Pydantic 构造）均在 `docs/source/api/` 有示例。

## 2. 目标包结构

```
habit/
  __init__.py          # 顶层聚合 + __version__ + is_available
  api/
    __init__.py        # 显式 re-export 表（文档生成入口）
    preprocessing.py   # 预处理 runner + Config
    habitat.py         # 分割 + 特征 + radiomics
    machine_learning.py
    analysis.py        # ICC, test-retest, compare
    components.py      # 可选：MSI, ITH 等小组件（B 档）
  exceptions.py        # B 档：HabitError 层次
  core/                # 内部实现（doc 标注为 private）
  utils/               # 部分 utils 公开（log, progress）
```

## 3. 顶层 `habit/__init__.py` 公开符号（A 档草案）

以下为 **A 档 MVP** 建议暴露的最小集合；B 档再扩展 `components`。

### 3.1 元信息

```python
__version__: str          # 与 pyproject.toml / setup.py 同步
__all__: list[str]        # 仅列稳定符号
```

### 3.2 Pipeline runners（路径 A）

| 公开名 | 实际实现 | CLI 命令 |
|--------|----------|----------|
| `run_preprocess` | `habit.core.preprocessing.run.run_preprocess_from_config` | `habit preprocess` |
| `run_dicom_sort` | `habit.core.dicom_sort.run.run_dicom_sort` | （prepare 流程） |
| `run_habitat_analysis` | `habit.core.habitat_analysis.run.run_habitat_analysis_from_config` | `habit get-habitat` |
| `run_feature_extraction` | `habit.core.habitat_analysis.run.run_feature_extraction_from_config` | `habit extract-features` |
| `run_radiomics` | `habit.core.habitat_analysis.run.run_radiomics_from_config` | `habit radiomics` |
| `run_ml` | `habit.core.machine_learning.run.run_ml_from_config` | `habit ml` |
| `run_kfold` | `habit.core.machine_learning.run.run_kfold_from_config` | `habit cv` |
| `run_model_comparison` | `habit.core.machine_learning.run.run_model_comparison_from_config` | `habit compare` |
| `run_icc_analysis` | `habit.core.machine_learning.feature_selectors.icc.icc.run_icc_analysis_from_config` | `habit icc` |

**命名说明**：公开名去掉 `_from_config` 后缀，缩短嵌入代码；内部函数名可保留别名 deprecated 一版。

### 3.3 Config 类（路径 C）

| 公开名 | 模块 |
|--------|------|
| `PreprocessingConfig` | `habit.core.preprocessing.config_schemas` |
| `DicomSortConfig` | `habit.core.dicom_sort` |
| `HabitatAnalysisConfig` | `habit.core.habitat_analysis.config_schemas` |
| `FeatureExtractionConfig` | 同上 |
| `MLConfig` | `habit.core.machine_learning.config_schemas` |
| `ModelComparisonConfig` | 同上 |
| `ICCConfig` | `habit.core.machine_learning.config_schemas` |
| `TestRetestConfig` | 同上 |

统一工厂方法：`XxxConfig.from_file(path: str | Path) -> XxxConfig`

### 3.4 CLI 覆盖 helper（可选公开）

| 公开名 | 用途 |
|--------|------|
| `apply_habitat_cli_overrides` | predict / debug / resume |
| `apply_ml_mode_override` | train / predict 切换 |

### 3.5 工具（有限公开）

| 公开名 | 模块 |
|--------|------|
| `setup_logger` | `habit.utils.log_utils` |
| `is_available` | 可选依赖探测（若已有则 re-export） |

### 3.6 B 档组件（路径 B）— `habit.api.components`

| 类 / 函数 | 用途 |
|-----------|------|
| `MSIFeatureExtractor` | MSI 矩阵与统计量 |
| `ITHScoreExtractor`（或实际类名） | ITH |
| `NonRadiomicsFeatureExtractor` | 形态统计 |
| `ClusteringAlgorithmFactory` | 聚类插件 |
| `HabitatFeatureRegistry` | habitat 特征插件 |
| `PreprocessorFactory` | 预处理插件 |
| `ModelFactory` | ML 模型插件 |

组件 API 需满足：

- 输入：`numpy.ndarray` / `SimpleITK.Image` / 明确 typed dict
- 输出：`pandas.DataFrame` 或 `dict[str, float]`
- 文档中有**不依赖 YAML** 的最小示例

## 4. 实现模式（lazy export 模板）

```python
# habit/__init__.py — illustrative sketch only
from __future__ import annotations
from typing import Any, Dict, Tuple
from habit.utils.lazy_exports import lazy_getattr

__version__ = "0.2.0"

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "run_preprocess": ("habit.api.preprocessing", "run_preprocess"),
    "PreprocessingConfig": ("habit.api.preprocessing", "PreprocessingConfig"),
    # ...
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    return lazy_getattr(name, globals(), _LAZY_EXPORTS)
```

`habit/api/preprocessing.py` 做薄 re-export，便于 Sphinx `automodule habit.api.preprocessing`。

## 5. 弃用策略（B 档起）

| 旧路径 | 新路径 | 时间线 |
|--------|--------|--------|
| `habit.core.preprocessing.run.run_preprocess_from_config` | `habit.run_preprocess` | 保留旧名 ≥2 minor，发 `DeprecationWarning` |
| 深路径 Config import | `habit.PreprocessingConfig` | 同上 |

实现：`habit.utils.deprecation.deprecated_alias` 装饰器或 import hook 文档说明。

## 6. 类型与签名规范

- 所有公开函数必须有完整 type hints（参数 + 返回值）。
- Config 对象优先于 `dict`；若接受 `dict`，仅作为 `model_validate` 前的便利，不在 runner 签名中暴露裸 dict。
- Logger 参数统一：`logger: logging.Logger | None = None`。
- 返回类型：与现有 runner 一致（如 `run_habitat_analysis` → `pd.DataFrame | None`），在 stub / docstring 中写清。

## 7. 公开 API 契约测试（A 档必做）

新文件 `tests/test_public_api.py`：

```python
import habit

EXPECTED = [
    "__version__",
    "run_preprocess",
    "PreprocessingConfig",
    # ... full __all__
]

def test_public_all_importable():
    for name in habit.__all__:
        getattr(habit, name)

def test_public_all_matches_documented():
    assert set(habit.__all__) == set(EXPECTED)
```

防止 accidental breaking：CI 中若删改 `__all__` 必须同步更新 `EXPECTED` 与 CHANGELOG。

## 8. 与 `docs/source/api/python_api.rst` 的迁移

升级后用户文档示例统一为：

```python
from habit import PreprocessingConfig, run_preprocess
from habit.utils.log_utils import setup_logger

config = PreprocessingConfig.from_file("config/preprocessing/config_preprocessing_demo.yaml")
run_preprocess(config, logger=setup_logger(...))
```

旧深路径示例保留一个 release 周期的「迁移说明」小节。

## 9. test-retest 公开面（待补齐）

当前 `run_test_retest` 仅在 `habit.commands.cmd_test_retest`，无 `run_*_from_config`。

**A 档选项**：

- **选项 1（推荐）**：新增 `habit.core.machine_learning.run.run_test_retest_from_config(config: TestRetestConfig) -> None`，CLI 与 `habit.run_test_retest` 共用。
- **选项 2**：A 档不暴露，文档标注「仅 CLI」，B 档再补。

## 10. 评审检查清单（实施前）

- [ ] `__all__` 列表与 README 公开 API 表一致
- [ ] 每个 runner 有对应 CLI parity 测试
- [ ] lazy import 不触发 radiomics import（`import habit` 耗时 < 200ms 无可选依赖环境）
- [ ] `habit.__version__` 与打包版本一致
- [ ] License 头文件模板不变
