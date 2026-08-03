# HABIT API 升级计划（子模块）

本目录是 HABIT 从「CLI 科研工具」升级为「可嵌入的优秀开源库」的**执行计划**，与 `docs/` 用户文档、`tests/` 测试代码分离，供维护者与 Agent 按阶段落地。

## 两个阶段

| 阶段 | 定位 | 状态 |
|------|------|------|
| **v0.1.x — API 门面** | API 是 CLI 的稳定门面，不重构业务逻辑 | ✅ 已完成并发布 |
| **v1.0 — API 优先** | API 是核心，CLI 退化为壳；目标是嵌入影像组学科研生态 | 🔵 设计评审中（分支 `v1.0.0`） |

## 文档索引

### v1.0（当前工作）

| 文件 | 内容 |
|------|------|
| [06_v1_api_first_architecture.md](06_v1_api_first_architecture.md) | **v1.0 架构设计**：六层结构、五个领域协议、`HabitatModel` 可流通、两级执行契约、YAML↔Python 双向同构、实施路线 |
| [07_v1_refactor_plan_and_usage.md](07_v1_refactor_plan_and_usage.md) | **v1.0 执行计划 + 重构后使用手册**：许可证迁移记录、阶段 0～7 的任务与验收、CLI 全部 16 个命令、API 各层与 70+ 内置组件、扩展开发与迁移指南 |
| [prototype/](prototype/) | 关键接口原型（仅签名与契约，不参与打包）：`contracts.py`、`protocols.py`、`spec.py`、`usage_examples.py` |

### v0.1.x（历史记录）

| 文件 | 内容 | 备注 |
|------|------|------|
| [01_master_plan.md](01_master_plan.md) | 目标标准、现状审计、三档升级路径 | 其中「API 是 CLI 门面、不重构业务逻辑」的定位已被 `06` 取代 |
| [02_public_api_design.md](02_public_api_design.md) | 顶层 `import habit` 公开面设计、`__all__` | 同上 |
| [03_testing_strategy.md](03_testing_strategy.md) | 五层 API 测试策略、golden 测试、契约测试 | 仍适用，v1.0 在此基础上扩展 |
| [04_ci_quality_governance.md](04_ci_quality_governance.md) | CI 矩阵、类型标注、版本/弃用、文档与治理 | 仍适用 |
| [05_roadmap_and_estimates.md](05_roadmap_and_estimates.md) | 分 PR 路线图、验收标准、估算 | v1.0 路线见 `06` 第 12 节 |

## 使用方式

1. **设计评审**：读 `06_v1_api_first_architecture.md`，对照 `prototype/usage_examples.py` 看调用形态。
2. **实施顺序**：按 `07_v1_refactor_plan_and_usage.md` 第一部分推进，阶段 0（golden 基线）必须先完成且只能在本地跑（`demo_data/` 未纳入 git）。
3. **交付形态**：`07` 第二部分即重构后的用户文档草稿，阶段 7 时并入 `docs/`。
4. **测试**：`03_testing_strategy.md` 与 `tests/` 同步扩展；架构分层规则由 `tests/test_architecture_contracts.py` 强制。

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
| v0.1.x — 公开 API 门面 | ✅ 已完成 |
| v0.1.x — golden / smoke / parity 测试 | ✅ 已完成 |
| v0.1.x — CI 基础门禁 | ✅ 已完成 |
| v0.1.x — 文档与 CHANGELOG | ✅ 已完成 |
| **v1.0 — 架构设计与接口原型** | ✅ 已完成（`06` + `prototype/`） |
| **v1.0 — 执行计划与使用手册** | ✅ 已完成（`07`） |
| **v1.0 — 许可证迁移至 Apache-2.0** | ✅ 已完成 |
| v1.0 — 阶段 0：golden 基线固化 | ⬜ 待开始 |
| v1.0 — 阶段 1：L2 契约层 | ⬜ 待开始 |
| v1.0 — 阶段 2：生境分割垂直切片 | ⬜ 待开始 |
| v1.0 — 阶段 3～6 | ⬜ 待开始 |
