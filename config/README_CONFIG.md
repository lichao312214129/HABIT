# 配置文件说明 / Configuration Files Guide

## 📋 可用配置文件 / Available Configuration Files

| 配置文件 | 功能 | 中文详解 | English Guide |
|---------|------|---------|---------------|
| `config_getting_habitat.yaml` | Habitat分析 | [📖 中文](config_getting_habitat_CN.yaml) | [📖 EN](config_getting_habitat_EN.yaml) |
| `config_machine_learning.yaml` | 机器学习 | [📖 中文](config_machine_learning_CN.yaml) | [📖 EN](config_machine_learning_EN.yaml) |
| `config_machine_learning_kfold.yaml` | K折交叉验证 | [📖 中文](config_machine_learning_kfold_CN.yaml) | [📖 EN](config_machine_learning_kfold_EN.yaml) |
| `config_image_preprocessing.yaml` | 图像预处理 | [📖 中文](config_image_preprocessing_CN.yaml) | [📖 EN](config_image_preprocessing_EN.yaml) |
| `config_extract_features.yaml` | 特征提取 | [📖 中文](config_extract_features_CN.yaml) | [📖 EN](config_extract_features_EN.yaml) |
| `config_model_comparison.yaml` | 模型比较 | [📖 中文](config_model_comparison_CN.yaml) | [📖 EN](config_model_comparison_EN.yaml) |
| `config_icc_analysis.yaml` | ICC分析 | [📖 中文](config_icc_analysis_CN.yaml) | [📖 EN](config_icc_analysis_EN.yaml) |

## ⚠️ 重要提示 / Important Notes

### YAML格式规范

1. **缩进**：
   - ✅ 使用**2个空格**进行缩进
   - ❌ **不要使用Tab键**
   - 保持层级关系清晰

2. **冒号**：
   - 冒号后面**必须有空格**: `key: value`
   - 如果值为空，可以不写或写 `null`

3. **列表**：
   - 使用 `-` 开头
   - `-` 后面**必须有空格**

4. **注释**：
   - 使用 `#` 开头
   - 可以单独一行或在行尾

5. **字符串**：
   - 一般不需要引号
   - 包含特殊字符时使用引号

### 示例 / Examples

```yaml
# ✅ 正确格式 / Correct Format
data_dir: ./data
output: ./results
settings:
  key1: value1
  key2: value2
  list:
    - item1
    - item2

# ❌ 错误格式 / Wrong Format
data_dir:./data                 # 冒号后缺少空格
output: ./results
settings:
    key1: value1                # 缩进用了4个空格（应该是2个）
  key2: value2                  # 缩进不一致
    list:
    -item1                      # 连字符后缺少空格
```

## 🔧 配置文件使用 / Configuration Usage

### CLI方式 / Using CLI

```bash
# 使用默认配置
habit habitat

# 使用指定配置文件
habit habitat --config config/config_getting_habitat.yaml

# 简写
habit habitat -c config/config_getting_habitat.yaml
```

### 脚本方式 / Using Scripts

```bash
python scripts/app_getting_habitat_map.py --config config/config_getting_habitat.yaml
```

## 📝 配置文件模板 / Configuration Templates

详细的配置文件模板请参考带 `_CN` 或 `_EN` 后缀的文件。

---

*最后更新 / Last Updated: 2025-10-19*

