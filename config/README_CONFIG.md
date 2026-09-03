# HABIT config templates

面向医生/临床用户的 YAML 模板：复制一份 → 只改标了 **★【必须修改】** 的字段 → 用 `habit check-config` 检查 → 再跑对应命令。

Relative paths in YAML resolve from the **config file directory**, not the shell working directory.

## 怎么选文件

| 用途 | 推荐先用 | 完整教学版 |
| --- | --- | --- |
| 日常只改路径就跑 | `*_minimal.yaml` | `*_demo.yaml` 或同名完整模板 |
| 想搞懂每个参数 | 完整版（含中英注释、按临床决策顺序分区） | — |

所有 YAML（包括专项模板、输入清单和 PyRadiomics 参数）均采用以下标识：

- **★【必须修改 / MUST EDIT】**：替换为本项目真实路径、列名、模态名或预测文件；不改通常无法用于真实数据。
- **★【必须检查 / MUST REVIEW】**：与真实数据结构逐项核对，例如固定模态、受试者 ID、mask 对应关系。
- **★【按研究方案确认 / STUDY-DESIGN REVIEW】**：会改变特征定义或统计结果；在研究开始前确定，同一研究中不要按结果临时调整。
- **常用**：初次可按模板运行，理解后再按研究方案调整。
- **高级**：并行、断点和诊断设置；首次使用请保持默认。

## YAML 注意（路径要不要加引号？）

- 用空格缩进，不要用 Tab；冒号后必须有空格：`key: value`
- `true` / `false` / `null` 小写
- **路径一般不用加引号**。仅当路径含空格或特殊字符（如 `#` `:` `*`）时才加引号  
  - `D:/data/images` → 可不加  
  - `"D:/my data/images"` → 有空格必须加  
- Windows 推荐正斜杠：`D:/work/...`
- 用记事本或 VS Code 保存为 `.yaml`，不要用 Word

改完可先检查（不跑完整流程）：

```powershell
habit check-config -c config/preprocessing/config_preprocessing_minimal.yaml
# 或显式指定工作流：
habit check-config -c config/machine_learning/config_machine_learning_kfold_minimal.yaml -w cv
# 输入清单或 PyRadiomics 参数预设只检查 YAML 格式：
habit check-config -c config/radiomics/parameter.yaml --syntax-only
```

## Catalog by workflow

| Workflow | Directory | Typical command |
| --- | --- | --- |
| Preprocessing | `preprocessing/` | `habit preprocess -c …` |
| DICOM sort | `dicom_sort/` | `habit sort-dicom -c …` |
| Habitat segmentation | `habitat/` | `habit get-habitat -c …` |
| Habitat feature extraction | `feature_extraction/` | `habit extract -c …` |
| Traditional radiomics | `radiomics/` | `habit radiomics -c …` |
| Machine learning | `machine_learning/` | `habit model -c …` / `habit cv -c …` |
| Model comparison | `model_comparison/` | `habit compare -c …` |
| ICC | `auxiliary/` | `habit icc -c …` |

## Starter demos（完整注释）与 minimal（只改 ★）

| Workflow | Demo（教学） | Minimal（日常） |
| --- | --- | --- |
| Preprocessing | `preprocessing/config_preprocessing_demo.yaml` | `…/config_preprocessing_minimal.yaml` |
| Habitat | `habitat/config_habitat_two_step.yaml` | `…/config_habitat_two_step_minimal.yaml` |
| Feature extraction | `feature_extraction/config_extract_features_demo.yaml` | `…/config_extract_features_minimal.yaml` |
| Radiomics | `radiomics/config_traditional_radiomics.yaml` | `…/config_traditional_radiomics_minimal.yaml` |
| ML train | `machine_learning/config_machine_learning_radiomics.yaml` | `…/config_machine_learning_radiomics_minimal.yaml` |
| ML k-fold | `machine_learning/config_machine_learning_kfold_demo.yaml` | `…/config_machine_learning_kfold_minimal.yaml` |
| ML prediction | `machine_learning/config_machine_learning_predict.yaml` | 使用完整模板，必须替换 `pipeline_path` |
| Model comparison | `model_comparison/config_model_comparison_demo.yaml` | `…/config_model_comparison_minimal.yaml` |
| ICC | `auxiliary/config_icc_demo.yaml` | `…/config_icc_minimal.yaml` |

模板按医生完成研究的决策顺序分区：

0. **文件说明** — 目的、前置步骤、运行命令、YAML 防错与必改项速查
1. **必改 ★：数据、结局与输出** — 路径、模态、ROI、列名、训练/预测模型路径
2. **按研究方案确认 ★：分析设计与流程** — 影像处理顺序、特征定义、队列划分、聚类/筛选规则
3. **候选模型/统计方法或高级运行** — 多候选方法时说明其研究依据；否则放并行、断点和诊断设置
4. **结果与评估** — 导出文件、图表和临床指标；若存在，应放在文件最后

完整教学模板中，每个实际配置项应说明：**它做什么、为什么需要它、何时才应修改、
可接受的取值/单位，以及是否会改变研究结论**。最小模板保留必要提示，不重复完整解释。

仅顶层区块使用上下两条 `# =============================================================================`；二级模块使用一条
`# --- 2.1 临床动作 / yaml_key ---`。模型、列表项和普通字段不使用粗分隔线，避免让
技术细节与医生必须完成的任务具有相同视觉权重。不要使用 `#%%` 或其他旧分隔符。

## 专项模板与数据清单

- `preprocessing/files_preprocessing*.yaml`、`preprocessing/image_files.yaml`、`habitat/file_habitat*.yaml` 是输入清单；每个受试者 ID、模态键和对应路径均标有 ★，必须与主流程配置一致。使用 `habit check-config -c <清单.yaml> --syntax-only` 只检查 YAML 格式。
- `radiomics/parameter*.yaml`、`radiomics/params_*radiomics.yaml` 是 PyRadiomics 参数预设；它们不是每次运行都要改，但其中带 ★ 的滤波、特征类别、归一化和离散化设置必须在研究方案中预先确认。
- `machine_learning/config_machine_learning_predict.yaml` 只使用 `input` 中第一张表，并要求已训练的 `pipeline_path`。
- `model_comparison/config_model_comparison*.yaml` 的 `files_config` 需要逐个核对预测 CSV 的路径与列名；更换上游模型后尤其要核对 `prob_col` 与 `pred_col`。

## Full field reference

See the Sphinx configuration pages (recipe catalog and per-workflow field docs):

https://lichao312214129.github.io/HABIT/configuration/index.html
