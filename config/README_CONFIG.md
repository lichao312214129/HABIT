# Configuration Files Guide

## Overview

HABIT toolkit uses YAML configuration files to control all functionalities. Each module has **two versions**:

1. **Standard Version** (`config_xxx.yaml`) - Concise configuration for daily use
2. **Annotated Version** (`config_xxx_annotated.yaml`) - Detailed English comments explaining all parameters

## 📂 Configuration Files Overview

All annotated templates are located in the [`config_templates/`](../config_templates/) directory.

## Available Configuration Files

| Config File | Annotated Template | Module | Status |
|-------------|-------------------|--------|--------|
| `config_getting_habitat.yaml` | [`config_getting_habitat_annotated.yaml`](../config_templates/config_getting_habitat_annotated.yaml) | Habitat Analysis | ✅ Complete |
| `config_extract_features.yaml` | [`config_extract_features_annotated.yaml`](../config_templates/config_extract_features_annotated.yaml) | Feature Extraction | ✅ Complete |
| `config_machine_learning.yaml` | [`config_machine_learning_annotated.yaml`](../config_templates/config_machine_learning_annotated.yaml) | Machine Learning | ✅ Complete |
| `config_machine_learning_kfold.yaml` | [`config_machine_learning_kfold_annotated.yaml`](../config_templates/config_machine_learning_kfold_annotated.yaml) | K-Fold Cross-Validation | ✅ Complete |
| `config_model_comparison.yaml` | [`config_model_comparison_annotated.yaml`](../config_templates/config_model_comparison_annotated.yaml) | Model Comparison | ✅ Complete |
| `config_icc_analysis.yaml` | [`config_icc_analysis_annotated.yaml`](../config_templates/config_icc_analysis_annotated.yaml) | ICC Analysis | ✅ Complete |
| `config_image_preprocessing.yaml` | [`config_image_preprocessing_annotated.yaml`](../config_templates/config_image_preprocessing_annotated.yaml) | Image Preprocessing | ✅ Complete |
| `config_traditional_radiomics.yaml` | [`config_traditional_radiomics_annotated.yaml`](../config_templates/config_traditional_radiomics_annotated.yaml) | Traditional Radiomics | ✅ Complete |
| `config_habitat_test_retest.yaml` | `config_habitat_test_retest_annotated.yaml` | Test-Retest Mapping | 🔄 Coming Soon |
| `config_image_preprocessing_dcm2nii.yaml` | `config_image_preprocessing_dcm2nii_annotated.yaml` | DICOM Conversion | 🔄 Coming Soon |

## 💡 How to Use

### For Quick Start
Use the **standard version** (e.g., `config_getting_habitat.yaml`) for quick execution.

### For Learning & Customization
Refer to the **annotated template** in `config_templates/` (e.g., `config_templates/config_getting_habitat_annotated.yaml`) to:
- Understand each parameter's purpose and options
- See usage examples and recommendations
- Learn valid value ranges and default settings
- Get tips on parameter tuning

## ⚠️ YAML Format Specification

**Important formatting rules**:

1. **Indentation**:
   - ✅ Use **2 spaces** (DO NOT use Tab)
   - ❌ Never use Tab key
   - Keep hierarchy clear

2. **Colon**:
   - Space **required** after colon: `key: value`
   - Empty values can be omitted or set to `null`

3. **Lists**:
   - Start with `-` symbol
   - Space **required** after `-`

4. **Comments**:
   - Start with `#` symbol
   - Can be on separate line or at line end

5. **Strings**:
   - Usually no quotes needed
   - Use quotes when containing special characters

### Example

```yaml
# ✅ Correct Format
data_dir: ./data
output: ./results
settings:
  key1: value1
  key2: value2
  list:
    - item1
    - item2

# ❌ Wrong Format
data_dir:./data                 # Missing space after colon
output: ./results
settings:
    key1: value1                # Wrong indentation (4 spaces instead of 2)
  key2: value2                  # Inconsistent indentation
    list:
    -item1                      # Missing space after dash
```

## 🔧 Configuration Usage

### Using CLI (Recommended)

```bash
# Use default configuration
habit habitat

# Use specified configuration file
habit habitat --config config/config_getting_habitat.yaml

# Short form
habit habitat -c config/config_getting_habitat.yaml
```

### Using Scripts (Legacy)

```bash
python scripts/app_getting_habitat_map.py --config config/config_getting_habitat.yaml
```

## 📝 Creating New Annotated Files

When creating new annotated configuration files:

1. **Use Template**: Refer to [`config_templates/config_getting_habitat_annotated.yaml`](../config_templates/config_getting_habitat_annotated.yaml) as template
2. **Follow Structure**: Include headers, sections with clear dividers, detailed comments
3. **Document Parameters**: For each parameter, include:
   - Purpose and function
   - Valid options/values
   - Default values
   - Usage examples
   - Tips and warnings
4. **Update This File**: Add entry to the table above after completion

## 📚 Related Documentation

- **Main README**: [README.md](../README.md) / [README_en.md](../README_en.md)
- **Habitat Analysis**: [doc/app_habitat_analysis.md](../doc/app_habitat_analysis.md) / [doc_en/app_habitat_analysis.md](../doc_en/app_habitat_analysis.md)
- **Machine Learning**: [doc/app_of_machine_learning.md](../doc/app_of_machine_learning.md) / [doc_en/app_of_machine_learning.md](../doc_en/app_of_machine_learning.md)

---

*Last Updated: 2025-10-19*
