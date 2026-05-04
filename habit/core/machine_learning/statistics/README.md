# statistics 子模块说明

`statistics` 是机器学习模块的“统计检验层”，提供评估阶段需要的统计学检验能力，用于模型性能差异与校准质量判断。

## 为什么这个模块有必要

指标数值本身只能说明“谁高谁低”，但不能说明差异是否具有统计学意义。
`statistics` 提供显著性检验能力，避免只凭单次指标做不可靠结论。

## 主要职责

1. **AUC 差异显著性检验**
   - 提供 DeLong 检验能力，用于比较多个模型 AUC 差异是否显著。

2. **校准相关统计检验**
   - 提供 Hosmer-Lemeshow 检验、Spiegelhalter Z 检验等能力，支持概率校准分析。

3. **与评估流程集成**
   - 被 `evaluation` / comparison 工作流调用，不直接负责报告导出。

## 它是怎么起作用的

1. `evaluation` 层准备标签与预测概率数组。
2. 调用 `statistics` 中对应检验函数（DeLong、Hosmer-Lemeshow、Spiegelhalter Z）。
3. 返回统计量和 `p` 值。
4. 上层再将检验结果写入比较报告。

## 具体例子

### 例子 1：两个模型 AUC 显著性比较

- 输入：同一测试集上的 `y_true`、模型 A 的概率、模型 B 的概率。
- 调 `delong_test.py` 中实现计算 `p` 值。
- 若 `p < 0.05`，可认为 AUC 差异具有统计学意义（具体阈值按研究方案设定）。

### 例子 2：概率校准检验

- 输入：`y_true` 与预测概率。
- 调 `hosmer_lemeshow_test.py` 或 `spiegelhalter_z_test.py` 评估校准质量。

## 关键文件与作用

- `delong_test.py`
  - 实现 DeLong 相关统计计算。

- `hosmer_lemeshow_test.py`
  - 实现 Hosmer-Lemeshow 拟合优度检验。

- `spiegelhalter_z_test.py`
  - 实现 Spiegelhalter Z 校准检验。

## 输入与输出边界

- **输入**：标签数组、预测概率数组（或模型对比数组）。
- **输出**：统计量、p 值、可用于上层导出与解释的结构化结果。

## 与其他子模块关系

- 作为底层工具供 `evaluation` 和 comparison 流程使用。
- 不应直接依赖 workflow、runner、reporting。

## 维护建议

- 优先保证统计公式与边界条件处理的正确性（样本量、小样本、异常值）。
- 新增检验方法时，建议保持统一输入输出格式，便于上层无痛接入。
- 统计方法更新要配合回归测试，避免影响历史分析结论。
