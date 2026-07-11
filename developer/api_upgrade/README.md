# HABIT API 升级计划（子模块）

本目录是 HABIT 从「CLI 科研工具」升级为「可嵌入的优秀开源库」的**执行计划**，与 `docs/` 用户文档、`tests/` 测试代码分离，供维护者与 Agent 按阶段落地。

## 文档索引

| 文件 | 内容 |
|------|------|
| [01_master_plan.md](01_master_plan.md) | 目标标准、现状审计、三档升级路径（MVP / 目标 / 满配） |
| [02_public_api_design.md](02_public_api_design.md) | 顶层 `import habit` 公开面设计、`__all__`、三种用法路径 |
| [03_testing_strategy.md](03_testing_strategy.md) | 五层 API 测试策略、golden 测试、契约测试 |
| [04_ci_quality_governance.md](04_ci_quality_governance.md) | CI 矩阵、类型标注、版本/弃用、文档与治理 |
| [05_roadmap_and_estimates.md](05_roadmap_and_estimates.md) | 分 PR 路线图、验收标准、LOC 与 token 估算 |

## 使用方式

1. **决策**：先读 `01_master_plan.md`，选定档位（建议从 **A 档 MVP** 开始）。
2. **设计评审**：`02_public_api_design.md` 需维护者确认后再改 `habit/__init__.py`。
3. **实施**：按 `05_roadmap_and_estimates.md` 的 PR 顺序推进；每 PR 对照对应文件的验收标准。
4. **测试**：`03_testing_strategy.md` 与 `tests/integration/test_python_api.py` 同步扩展。

## 基线数据（2026-07-06 统计）

| 指标 | 数值 |
|------|------|
| `habit/` Python 文件 | 281 |
| `habit/` 源码行数 | ~52,900 |
| 顶层 `class` / `def` | ~909 |
| `tests/` 文件 / 行数 | 62 / ~2,743 |
| `docs/` `.rst` 文件 | 58 |
| GitHub Actions | 仅 `docs.yml`，无 pytest CI |

## 与现有文档的关系

- 用户向 API 示例：`docs/source/api/python_api.rst`（升级后需与 `02_public_api_design.md` 对齐）。
- 架构契约：`tests/test_architecture_contracts.py`（registry / orchestrator，已存在 sklearn 风格 `check_*` 思路）。
- 并行可靠性：`docs/HABITAT_PARALLEL_RELIABILITY_PLAN.md`（与 API 升级正交，可并行）。

## 状态

| 阶段 | 状态 |
|------|------|
| 计划编写 | ✅ 已完成 |
| A 档 MVP — 公开 API 门面 | ✅ 已完成 |
| A 档 MVP — golden / smoke / parity 测试 | ✅ 已完成 |
| A 档 MVP — CI 基础门禁 | ✅ 已完成 |
| A 档 MVP — 文档与 CHANGELOG | ✅ 已完成 |
| B 档目标实施 | ⬜ 待开始 |
| C 档满配 | ⬜ 待开始 |
