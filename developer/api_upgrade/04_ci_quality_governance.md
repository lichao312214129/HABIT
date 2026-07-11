# 04 — CI、质量与治理

## 1. CI 现状与目标

### 现状

- `.github/workflows/docs.yml` — 仅构建 Sphinx 文档
- `.pre-commit-config.yaml` — black / mypy / pylint，**未接入 CI**
- 无 pytest workflow；无覆盖率上报

### A 档 CI 目标

新文件：`.github/workflows/tests.yml`

```yaml
# 结构示意 — 实施时写完整 YAML
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt && pip install -e .
      - run: pytest tests/test_public_api.py tests/integration/ tests/api/ -m "not slow" -v
```

### B 档 CI 矩阵

| OS | Python | 说明 |
|----|--------|------|
| ubuntu-latest | 3.10, 3.11 | 主矩阵 |
| windows-latest | 3.10 | 路径与 spawn 差异 |
| macos-latest | 3.10 | 可选 |

可选依赖分组：

```ini
# pytest 分组安装
pip install -r requirements.txt
pip install -e .
# extras: radiomics, gpu — 单独 job 或 continue-on-error
```

## 2. 静态检查门禁

| 工具 | A 档 | B 档 |
|------|------|------|
| black | CI `black --check habit tests` | 同 |
| mypy | 仅 `habit/api/` + 公开 runner 签名 | 全 `habit/` |
| pylint | 不阻塞 | 警告级 |
| pre-commit | 文档推荐 | CI 同步 |

`pyproject.toml` 需统一：

- 单一 build-backend（setuptools 或 poetry，二选一）
- 锁定最低依赖版本（`numpy>=1.23` 等），避免 `numpy = "*"`

## 3. 版本管理

### 3.1 单一版本源

```
habit/_version.py          # __version__ = "0.2.0"
pyproject.toml             # dynamic version 或同步脚本
setup.py                   # 读 _version.py
```

CI 检查：版本三处一致脚本 `developer/api_upgrade/scripts/check_version_sync.py`（A 档可选）。

### 3.2 SemVer 规则

| 变更类型 | 版本 bump |
|----------|-----------|
| 删除/改名公开 `__all__` 符号 | MAJOR |
| 新增公开 API | MINOR |
| 修复 bug、文档、内部重构 | PATCH |
| 弃用（仍可用 + Warning） | MINOR |

### 3.3 CHANGELOG

根目录 `CHANGELOG.md`，遵循 [Keep a Changelog](https://keepachangelog.com/)：

```markdown
## [Unreleased]
### Added
- Public API: `habit.run_preprocess`, ...

### Deprecated
- `habit.core.preprocessing.run.run_preprocess_from_config` → use `habit.run_preprocess`
```

与 `docs/source/changelog.rst` 交叉链接，避免双份维护：rst 可 `.. include::` 或仅链接 GitHub CHANGELOG。

## 4. 弃用机制（B 档）

`habit/utils/deprecation.py`：

```python
def deprecated(since: str, alternative: str):
    """Emit DeprecationWarning with stacklevel=2."""
```

公开 runner 旧名保留 wrapper 一版。

## 5. 异常体系（B 档）

```
habit/exceptions.py
  HabitError                 # base
  ConfigValidationError      # Pydantic 包装
  PipelineError              # habitat pipeline
  OptionalDependencyError    # radiomics/torch 缺失
```

Runner 边界捕获并 re-raise 为 `HabitError` 子类，避免裸 `Exception`。

## 6. 文档工程

### A 档

- 更新 `docs/source/api/python_api.rst` 使用顶层 import
- 新增「Public API stability」小节在 `docs/source/development/` 或链接本目录

### B 档

- Sphinx autodoc：`habit.api`, `habit.exceptions`
- numpydoc 模板；公开函数 docstring 必填 Parameters / Returns / Examples
- `examples/` 目录：

```
examples/
  01_preprocess_api.py
  02_habitat_train_api.py
  03_msi_component.py
  README.md
```

- Gallery：sphinx-gallery 或手动 rst literalinclude

## 7. 社区与治理（B–C 档）

| 文件 | 用途 |
|------|------|
| `.github/PULL_REQUEST_TEMPLATE.md` | PR 检查 API / CHANGELOG |
| `.github/ISSUE_TEMPLATE/bug_report.md` | |
| `CONTRIBUTING.md` 或链到 `docs/.../contributing.rst` | |
| `GOVERNANCE.md` | 维护者、发布流程 |
| `ROADMAP.md` | 公开路线图（可摘要本计划 A/B 档） |

## 8. License 战略（决策项）

| 选项 | 生态影响 |
|------|----------|
| 维持当前非商业 License | 学术可用；企业集成需单独授权 |
| 双许可（非商业 + 商业授权） | 常见国内开源模式 |
| BSD / MIT | 最大化生态采用；与 sklearn 同级 |

**本 API 升级计划不强制改 License**，但在 B 档「生态重要成员」目标下需在 `01_master_plan.md` 决策记录中注明结论。

## 9. 打包与发布（B 档）

- PyPI 包名确认（当前 poetry name `HABIT`）
- `pip install habit`  smoke test job
- Tag 发布：`v0.2.0` → GitHub Release + PyPI

## 10. CI 验收标准（A 档）

- [ ] PR 触发 pytest（unit + 非 slow）
- [ ] main 分支 docs + tests 均绿
- [ ] black check 通过
- [ ] README 徽章：tests passing（可选）
