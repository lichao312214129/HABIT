# 01 — 总目标与升级路径

## 1. 愿景

HABIT 应成为**影像组学生态中可嵌入的基础设施**：用户与第三方开发者能够像使用 `sklearn`、`pyradiomics`、`MONAI` 一样，通过稳定的 Python API 将 habitat 分割、特征提取、机器学习流程集成到自己的 notebook、流水线或产品中。

**成功标志（可观测）**：

- `pip install habit` 后 `import habit` 即可获得文档化、 semver 保证的公开 API。
- 核心算法有**数值 golden 测试**，CI 全绿才允许合并。
- 官方示例「复制即可运行」，不依赖深路径 `habit.core.*.run`。
- 第三方仓库能在 issue/论文中引用 HABIT 版本号并复现结果。

## 2. 「优秀开源库」的七维标准

| 维度 | sklearn / 生态标杆 | HABIT 当前 | 目标 |
|------|-------------------|------------|------|
| **1. 公开 API 契约** | `from sklearn.xxx import Yyy`，`__all__` 清晰 | 顶层 `habit/__init__.py` 几乎为空；用户需深路径导入 | 扁平稳定入口 + lazy export |
| **2. 版本与稳定性** | SemVer + DeprecationWarning + CHANGELOG | 0.1.0，无弃用政策 | 1.0.0 前冻结公开面；弃用至少保留 2 个小版本 |
| **3. 测试深度** | 数值正确性 + 高覆盖率 + CI 门禁 | 大量手工脚本；API 测试多 mock，不测数值 | golden + 组件 + 契约 + 可选 E2E |
| **4. CI/CD** | 多 OS × 多 Python + lint + coverage | 仅 docs 构建 | pytest + mypy + pre-commit 进 CI |
| **5. 文档三件套** | User Guide + API Reference + Gallery | Sphinx 用户文档较好；缺 autodoc API 与 gallery | numpydoc + autodoc + examples/ |
| **6. 可嵌入性** | numpy/pandas 标准 I/O；Estimator 接口 | 深度绑定 YAML Config；组件独立性未保证 | Config 路径 + 组件路径双轨；可选 sklearn mixin |
| **7. 社区治理** | BSD/MIT + CONTRIBUTING + 模板 | 自定义 License（非商业限制） | 治理文件齐全；License 战略需决策 |

## 3. 现状审计（摘要）

### 3.1 已有优势

- **域模块已有 lazy export**：`habit.core.habitat_analysis`、`habit.core.machine_learning` 等已实现 `__getattr__` + `__all__`。
- **Runner 模式清晰**：各域 `run_*_from_config` 与 CLI 薄封装一致（见 `habit/commands/`）。
- **Pydantic Config**：`XxxConfig.from_file` / `model_validate` 类型安全。
- **架构契约测试**：`tests/test_architecture_contracts.py` 检查 registry / orchestrator。
- **Sphinx 文档**：配置、CLI、特征定义已做代码一致性审计。

### 3.2 关键缺口

1. **顶层包未聚合公开面** — `import habit` 无可用符号。
2. **API 测试不验证计算** — `tests/integration/test_python_api.py` 以 mock 为主。
3. **无功能 CI** — 回归依赖本地手动跑脚本。
4. **深路径导入** — 文档示例写 `habit.core.habitat_analysis.run`，对嵌入者不友好。
5. **License 限制** — 非商业条款阻碍企业级生态采用（战略决策项）。

### 3.3 三种 API 用法路径（均需测试覆盖）

| 路径 | 典型用户 | 入口 | 现状 |
|------|----------|------|------|
| **A. Pipeline** | 跑完整工作流 | `run_*_from_config(config)` | 已实现，缺顶层 re-export |
| **B. Component** | 嵌入自定义流水线 | `MSIFeatureExtractor`、`ClusteringAlgorithmFactory` 等 | 分散在各子包，无稳定顶层 |
| **C. Config 编程** | 无 YAML 的 notebook | `HabitatAnalysisConfig(...)` 构造 | schema 支持，文档示例少 |

## 4. 三档升级路径

### A 档 — MVP「可被安全嵌入」

**目标**：第三方敢写 `habit>=0.2.0` 进 requirements。

| 工作项 | 交付物 |
|--------|--------|
| 公开 API 契约 | `habit/__init__.py` + `habit.api` 门面；`tests/test_public_api.py` |
| 核心 runner 测试 | 扩展 `test_python_api.py`：真实小规模 demo 数据 smoke |
| 算法 golden（核心） | MSI、ITH、non_radiomics 固定输入 → 固定输出 hash |
| 精简 CI | `.github/workflows/tests.yml`：Linux + py310 + pytest unit/integration |
| 版本与 CHANGELOG | `habit.__version__`；`CHANGELOG.md` 骨架 |

**规模**：~6,000–8,000 行新增/改动；Agent token ~10–15M。

### B 档 — 目标「优秀开源库」

在 A 档基础上：

| 工作项 | 交付物 |
|--------|--------|
| 覆盖率 ~85% | pytest-cov 门禁；habitat_analysis + ML 核心路径 |
| 全量类型标注 | mypy strict 对 `habit/` 公开 API |
| sklearn 兼容层 | 可选 `HabitatClusterer(BaseEstimator)` 等 |
| 自定义异常体系 | `habit.exceptions` |
| autodoc API Reference | Sphinx `automodule` 从 docstring 生成 |
| examples/ 画廊 | 5–10 个可运行 notebook/script |
| CI 矩阵 | Linux/Windows × py310/py311 |

**规模**：~20,000–30,000 行；token ~30–50M。

### C 档 — 满配「生态基础设施」

在 B 档基础上：>90% 覆盖率、全量 numpydoc、多版本长期维护、治理与路线图公开。这是**持续 6–18 个月**的维护投入，非一次性工程。

**规模**：~35,000–55,000 行；token ~60–100M+。

## 5. 推荐决策

1. **先做 A 档**，A 是 B/C 的地基；公开 API + CI + 核心 golden 投入产出比最高。
2. **License** 若目标是「中国最重要的开源包」，需在 B 档前评估是否调整为 BSD/MIT 或双许可。
3. **不重构业务逻辑** — 升级以「暴露 + 测试 + 门禁」为主，避免大规模重写 5.3 万行核心代码。

## 6. 非目标（本计划不做）

- 重写 habitat 分割算法或 ML 训练逻辑。
- 替换 YAML 配置体系（Config 仍是主路径，仅增加编程式构造文档）。
- GUI（`habit-gui`）API 化。
- 一次性达到 sklearn 全库 90%+ 覆盖率。
