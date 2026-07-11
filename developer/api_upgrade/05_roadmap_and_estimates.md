# 05 — 路线图、PR 清单与规模估算

## 1. 实施顺序总览

```
Phase 0  计划评审（本目录） ──► 你确认 A 档范围
Phase 1  公开 API 门面          PR-1 ~ PR-3
Phase 2  契约 + golden 测试     PR-4 ~ PR-7
Phase 3  CI 门禁               PR-8
Phase 4  文档与版本            PR-9 ~ PR-10
────────── A 档 MVP 完成 ──────────
Phase 5  B 档：覆盖率 + autodoc + examples   PR-11 ~
Phase 6  C 档：长期维护
```

## 2. A 档 PR 清单（建议）

### PR-1：版本源与 `habit/_version.py`

| 项 | 内容 |
|----|------|
| 改动 | 新增 `_version.py`；setup/pyproject 读取 |
| 文件 | ~3 文件，~40 行 |
| 验收 | `import habit; habit.__version__` 可用 |

### PR-2：`habit/api/` 门面模块

| 项 | 内容 |
|----|------|
| 改动 | `habit/api/preprocessing.py`, `habitat.py`, `machine_learning.py`, `analysis.py` |
| 文件 | ~5 文件，~200 行 |
| 验收 | 各模块 `__all__` 可导入；lazy 不拉 radiomics |

### PR-3：顶层 `habit/__init__.py` 聚合

| 项 | 内容 |
|----|------|
| 改动 | lazy export 全公开符号；更新 `02_public_api_design.md` 若有偏差 |
| 文件 | ~2 文件，~120 行 |
| 验收 | `tests/test_public_api.py` 全绿 |

### PR-4：扩展 mock 契约测试

| 项 | 内容 |
|----|------|
| 改动 | 补全 `test_python_api.py`：radiomics, compare, icc config load |
| 文件 | ~1 文件，~150 行 |
| 验收 | 覆盖全部 `run_*` runner 委托 |

### PR-5：MSI + ITH golden 测试

| 项 | 内容 |
|----|------|
| 改动 | fixtures + `tests/api/test_component_msi.py`, `test_component_ith.py` |
| 文件 | ~6 文件，~800 行（含小 fixture） |
| 验收 | golden 与当前实现一致；改算法需显式更新 snapshot |

### PR-6：Pipeline smoke（preprocess + extract）

| 项 | 内容 |
|----|------|
| 改动 | demo config + `@pytest.mark.integration` |
| 文件 | ~3 文件，~400 行 |
| 验收 | 本地 py310 2 分钟内跑完 |

### PR-7：CLI–API parity（preprocess）

| 项 | 内容 |
|----|------|
| 改动 | subprocess 对比输出目录 |
| 文件 | ~2 文件，~250 行 |
| 验收 | 除 log 外输出一致 |

### PR-8：GitHub Actions `tests.yml`

| 项 | 内容 |
|----|------|
| 改动 | pytest + black check |
| 文件 | ~2 文件，~80 行 |
| 验收 | PR 上 CI 绿 |

### PR-9：文档迁移 `python_api.rst`

| 项 | 内容 |
|----|------|
| 改动 | 顶层 import 示例；迁移说明 |
| 文件 | ~2 rst，~100 行 |
| 验收 | docs build 通过 |

### PR-10：`CHANGELOG.md` + 弃用说明占位

| 项 | 内容 |
|----|------|
| 改动 | Unreleased 条目 |
| 文件 | ~2 文件，~60 行 |
| 验收 | 与 PR-1~3 公开符号一致 |

**A 档合计**：约 10 PR，~2,100 行新增（含测试与 fixture），核心生产代码 ~500 行。

---

## 3. B 档 PR 概要（PR-11 起）

| PR | 内容 | 估行 |
|----|------|------|
| PR-11 | non_radiomics + clustering golden | ~1,200 |
| PR-12 | pytest-cov 85% 门禁 + 补测 ML/habitat | ~5,000 |
| PR-13 | mypy 全 habit + fixes | ~3,000（含类型修复） |
| PR-14 | `habit.exceptions` + runner 包装 | ~600 |
| PR-15 | `deprecated()` + 旧路径 alias | ~400 |
| PR-16 | Sphinx autodoc habit.api | ~500 |
| PR-17 | examples/ 5 脚本 + rst | ~800 |
| PR-18 | CI 矩阵 win + py311 | ~150 |
| PR-19 | `run_test_retest_from_config` 公开 | ~200 |
| PR-20 | sklearn `BaseEstimator` 试点（1 组件） | ~400 |

**B 档增量**：~12,000–18,000 行（大量为测试与类型修复）。

---

## 4. LOC 汇总表

| 档位 | 新增/改动 LOC | 其中测试+fixture | 其中生产代码 |
|------|---------------|------------------|--------------|
| A 档 MVP | 6,000–8,000 | ~5,000 | ~800–1,200 |
| B 档目标 | +14,000–22,000 | ~10,000 | ~2,000–4,000 |
| C 档满配 | +15,000–25,000 | ~12,000 | ~3,000–5,000 |
| **累计 C 档** | **35,000–55,000** | **~27,000** | **~6,000–10,000** |

说明：HABIT 现有 ~52,900 行生产代码**不需重写**；升级主要是门面、测试、CI、文档。

---

## 5. Agent Token 估算

估算假设：Agent 式开发（读上下文 → 写 → pytest → 修），**每 1 行最终留存代码 ~1,000–2,000 token**；读 52k 行既有代码占额外 ~5–10M token（分散在多 PR）。

| 档位 | 新增 LOC | 估算 token | 约合对话轮次（粗算） |
|------|----------|------------|----------------------|
| A 档 | ~7,000 | **10–15M** | 15–25 个 focused session |
| B 档 | +18,000 | **+20–35M** | 30–50 session |
| C 档 | +20,000 | **+30–50M** | 40–60 session |
| **合计 C 档** | ~45,000 | **60–100M+** | 跨 3–6 个月 |

波动因素：±50%（返工、golden 争议、Windows CI 调试）。

---

## 6. 人力与时间（参考）

| 档位 | 全职维护者 | 兼职 + Agent |
|------|------------|--------------|
| A 档 | 1–2 周 | 2–4 周 |
| B 档 | 1–2 月 | 2–3 月 |
| C 档 | 持续 6–18 月 | 持续 |

---

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| golden 与论文数值不一致 | PR-5 前人工核对 MSI/ITH；文档引用 Wu 2018 |
| lazy import 循环依赖 | 沿用现有 `lazy_exports` 模式；`import habit` 单测 |
| demo 数据过大 CI 慢 | fixture 极小化；slow 标记 + nightly |
| 公开 API 过早冻结 | A 档标 0.2.0 beta；1.0.0 前允许 minor 调整 |
| License 阻碍采用 | 在 B 档前单独决策 recorded in README |

---

## 8. A 档完成定义（Definition of Done）

- [x] `import habit` 暴露 PR-3 全部 `__all__` 符号
- [x] `pytest -m "not slow"` CI 绿
- [x] MSI + ITH golden 存在
- [x] ≥1 pipeline smoke + ≥1 CLI–API parity
- [x] `docs/source/api/python_api.rst` 使用新 import
- [x] `CHANGELOG.md` 记录 Unreleased
- [x] 本目录 README 状态表更新为「A 档 MVP 已完成」

---

## 9. 下一步行动（B 档）

1. 覆盖率门禁、autodoc、`examples/` 画廊（见 B 档 PR 概要）。
2. 可选：补 `run_test_retest` 公开 runner（见 02 第 9 节）。
3. License 战略决策（见 04_ci_quality_governance.md）。
