# 03 — API 测试策略

## 1. 测试哲学

HABIT 的 API 测试目标不是「调用有没有发生」，而是：

1. **公开符号稳定** — `import habit` 契约不破。
2. **算得对** — 核心算法输出与 golden 一致或在容差内。
3. **CLI ≡ API** — 同一 config 走 CLI 与走 runner，结果文件一致。
4. **组件可独立** — 不建 YAML 也能跑 MSI/ITH 等。

现有 `tests/integration/test_python_api.py` 以 mock 为主，适合快速契约；**A 档起必须增加非 mock 层**。

## 2. 五层测试金字塔

```
                    ┌─────────────────┐
                    │ 5. E2E API      │  可选；demo 数据全链路
                    └────────┬────────┘
               ┌─────────────┴─────────────┐
               │ 4. CLI–API Parity         │  同一 config 输出 diff
               └─────────────┬─────────────┘
          ┌──────────────────┴──────────────────┐
          │ 3. Pipeline smoke（真实小数据）      │
          └──────────────────┬──────────────────┘
     ┌─────────────────────────┴─────────────────────────┐
     │ 2. Component golden（MSI, ITH, clustering…）       │
     └─────────────────────────┬─────────────────────────┘
┌──────────────────────────────┴──────────────────────────────┐
│ 1. Public API contract + architecture（mock 可保留）         │
└─────────────────────────────────────────────────────────────┘
```

## 3. 第一层 — 公开 API 契约

| 文件 | 内容 |
|------|------|
| `tests/test_public_api.py` | `__all__` 可导入、符号类型、版本存在 |
| `tests/integration/test_python_api.py` | 保留 mock runner 委托测试 |
| `tests/test_architecture_contracts.py` | 已有；registry / orchestrator |

**标记**：`@pytest.mark.unit`

## 4. 第二层 — 组件 golden 测试

### 4.1 优先级组件

| 组件 | 模块 | Golden 策略 |
|------|------|-------------|
| MSI | `msi_features.py` | 固定 3D label map → 矩阵 + 4 统计量 JSON snapshot |
| ITH | `ith_score.py` | 固定 map → `ith_score`, `num_habitats` |
| Non-radiomics | `basic_features.py` | 固定 map → volume_ratio 等 |
| 聚类选 k | `cluster_selection` | 固定特征矩阵 + random_state → k |

### 4.2 数据 fixture

```
tests/fixtures/api_golden/
  msi_label_map.nii.gz          # 或 npy + 小 SimpleITK 生成
  msi_expected.json
  ith_label_map.nii.gz
  ith_expected.json
```

生成 golden 时：

1. 用当前实现跑一遍，人工核对与论文/手工计算一致。
2. 写入 JSON；测试只比对 key 集合与数值（`pytest.approx`）。

### 4.3 示例测试结构

```python
@pytest.mark.unit
def test_msi_extractor_golden(msi_fixture):
    extractor = MSIFeatureExtractor(...)
    result: pd.DataFrame = extractor.extract(...)
    assert_frame_equal(result, expected, rtol=1e-5)
```

### 4.4 文件

| 新文件 | 说明 |
|--------|------|
| `tests/api/test_component_msi.py` | MSI golden |
| `tests/api/test_component_ith.py` | ITH golden |
| `tests/api/test_component_non_radiomics.py` | 形态特征 |
| `tests/conftest.py` | 扩展 golden fixture 路径 |

## 5. 第三层 — Pipeline smoke（真实小数据）

使用 `demo_data/` 子集（或 CI 可下载的 tiny cohort）：

| 测试 | Config | 断言 |
|------|--------|------|
| preprocess smoke | `config_preprocessing_demo.yaml` | 输出目录存在预期 nii |
| habitat train smoke | 最小 one_step + 1 subject | `habitats.csv` 行数 |
| extract features smoke | `config_extract_features_demo.yaml` | 预期 CSV 列名前缀 |
| ml train smoke | radiomics demo + 2 fold 缩小 | `model.pkl` 存在 |

**标记**：`@pytest.mark.integration` + `@pytest.mark.slow`

**CI 策略**：A 档仅在 Linux py310 跑；PR 可选 skip slow，main 分支 nightly 跑全量。

## 6. 第四层 — CLI–API Parity

```python
def test_preprocess_cli_api_parity(tmp_path, demo_config):
    api_out = tmp_path / "api"
    cli_out = tmp_path / "cli"
    # API path
    cfg = PreprocessingConfig.from_file(demo_config)
    cfg.out_dir = str(api_out)
    run_preprocess(cfg)
    # CLI path
    subprocess.run(["habit", "preprocess", "-c", ...], check=True)
    assert_dirs_equal(api_out, cli_out, ignore=["*.log"])
```

| 新文件 | 覆盖命令 |
|--------|----------|
| `tests/api/test_cli_api_parity_preprocess.py` | preprocess |
| `tests/api/test_cli_api_parity_habitat.py` | get-habitat（train 小数据） |
| `tests/api/test_cli_api_parity_ml.py` | ml train |

依赖：`tests/utils/test_subprocess_utils.py` 模式。

## 7. 第五层 — E2E API（B 档）

`tests/integration/workflow_*.py` 已存在 CLI 导向 E2E；B 档增加纯 Python 入口版本：

```python
def test_api_workflow_preprocess_to_ml():
    cfg_pre = PreprocessingConfig.from_file(...)
    run_preprocess(cfg_pre)
    cfg_hab = HabitatAnalysisConfig.from_file(...)
    run_habitat_analysis(cfg_hab)
    ...
```

不与 CLI E2E 重复断言逻辑，共用 `assert_artifact_contract` helper。

## 8. Mock 测试的保留范围

**继续 mock**：

- Runner 是否正确调用 `BatchProcessor.run()`（已有）。
- Config validation 边界（predict 无 pipeline_path）。
- CLI override helper 行为。

**禁止仅 mock**：

- MSI / ITH / 聚类数值。
- 特征 CSV 列名与 dtype。

## 9. pytest 标记与目录约定

```
tests/
  api/                    # 新增：API 专项
    test_public_api.py    # 或放 tests 根目录
    test_component_*.py
    test_cli_api_parity_*.py
  integration/
    test_python_api.py    # 保留 mock 层
  fixtures/
    api_golden/
```

`pyproject.toml` 或 `pytest.ini` 注册 markers：

```ini
markers =
    unit: fast, no IO
    integration: needs demo data
    slow: > 60s
    gpu: needs CUDA
```

## 10. 覆盖率目标

| 档位 | 行覆盖率 | 公开 API 模块 |
|------|----------|---------------|
| A 档 | ≥60% `habit/` | 100% 公开 runner 有 smoke |
| B 档 | ≥85% | 100% + components golden |
| C 档 | ≥90% | 全模块 |

工具：`pytest-cov` + Codecov（可选）。

## 11. 与现有测试脚本的关系

`tests/habitat/habitat_one_step_*.py` 等**保留**为开发手动回归脚本，不删；CI 以 `test_*.py` 为准。长期可将脚本逻辑迁入 parametrized integration 测试。

## 12. 验收标准（A 档测试）

- [ ] `pytest tests/test_public_api.py tests/integration/test_python_api.py` 全绿
- [ ] 至少 MSI + ITH 两个 component golden 测试
- [ ] 至少 1 个 pipeline smoke（preprocess 或 feature extraction）
- [ ] 至少 1 个 CLI–API parity（preprocess 优先）
- [ ] CI workflow 在 PR 上自动运行 unit + 非 slow integration
