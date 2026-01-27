# Configuration Templates Summary

## ✅ Completed Work

We have successfully created **detailed English-annotated configuration templates** for all major HABIT modules.

## 📊 Status Overview

| Module | Standard Config | Annotated Template | Status |
|--------|----------------|-------------------|--------|
| Habitat Analysis | `config_getting_habitat.yaml` | `config_getting_habitat_annotated.yaml` | ✅ Complete |
| Feature Extraction | `config_extract_features.yaml` | `config_extract_features_annotated.yaml` | ✅ Complete |
| Machine Learning | `config_machine_learning.yaml` | `config_machine_learning_annotated.yaml` | ✅ Complete |
| K-Fold CV | `config_machine_learning_kfold.yaml` | `config_machine_learning_kfold_annotated.yaml` | ✅ Complete |
| Model Comparison | `config_model_comparison.yaml` | `config_model_comparison_annotated.yaml` | ✅ Complete |
| ICC Analysis | `config_icc_analysis.yaml` | `config_icc_analysis_annotated.yaml` | ✅ Complete |
| Image Preprocessing | `config_image_preprocessing.yaml` | `config_image_preprocessing_annotated.yaml` | ✅ Complete |
| Traditional Radiomics | `config_traditional_radiomics.yaml` | `config_traditional_radiomics_annotated.yaml` | ✅ Complete |

## 📝 What's Included in Each Annotated Template

Each annotated configuration file contains:

### 1. Header Section
- File description
- YAML formatting rules (indentation, colons, comments)
- Quick start commands (CLI and script)
- Links to relevant documentation

### 2. Detailed Parameter Explanations
For every parameter:
- **Purpose**: What it does
- **Data type**: Expected value type
- **Options**: Valid values or ranges
- **Defaults**: Default values
- **Examples**: Usage examples
- **Tips**: Best practices and warnings

### 3. Additional Sections
- **Usage Examples**: Real-world configuration scenarios
- **Best Practices**: Recommendations for effective use
- **Common Issues**: Troubleshooting tips
- **Output Files**: Description of generated files
- **Get Help**: Links to documentation and resources

## 🎯 Design Philosophy

1. **Single Language**: English only (no bilingual comments)
2. **Comprehensive**: Every parameter explained
3. **Practical**: Real-world examples and use cases
4. **Self-Contained**: Users can understand configs without external docs
5. **Formatted**: Clear structure with section dividers

## 📖 Documentation Structure

```
config/
├── config_xxx.yaml                    # Standard concise config for daily use
├── config_xxx_annotated.yaml          # Detailed template with full annotations
└── README_CONFIG.md                   # Index of all configuration files
```

## 💡 How Users Should Use Them

### For Learning
- Read the annotated version to understand all options
- Learn parameter meanings, ranges, and effects
- Review examples and best practices

### For Daily Use
- Copy annotated template as starting point
- Customize for your specific needs
- Save as standard config (without verbose comments)

### For Reference
- Refer back when unsure about parameters
- Check examples for specific use cases
- Review best practices before production use

## 📈 Key Features

### 1. YAML Format Guidelines
Every template includes:
- ✅ 2-space indentation rule (no tabs)
- ✅ Space after colon requirement
- ✅ List syntax with "- " prefix
- ✅ Comment syntax with "#"

### 2. Parameter Documentation
- **Type information**: String, integer, float, boolean, list
- **Value ranges**: Min/max values, valid options
- **Conditional parameters**: When parameters are required
- **Dependencies**: Parameter interactions

### 3. Practical Examples
- Minimal configurations
- Full-featured configurations
- Use-case-specific configurations
- Multi-scenario examples

### 4. Troubleshooting Guidance
- Common errors and solutions
- Performance optimization tips
- Quality control recommendations
- Resource management advice

## ⚠️ Special Handling: Voxel-based GLCM Features

When using `voxel_radiomics()` method for voxel-level feature extraction, HABIT automatically handles GLCM (Gray Level Co-occurrence Matrix) features to prevent extraction failures.

### Background

Voxel-based extraction uses small local neighborhoods (e.g., 3×3×3 or 5×5×5). In such small regions:
- Some GLCM features may fail due to overly homogeneous local textures
- PyRadiomics has strict requirements for feature names

### Automatic Protection Mechanism

HABIT uses PyRadiomics API to automatically restrict GLCM to safe features:

1. **When**: Detects when `glcm` is enabled in parameter file without specifying exact features
2. **Action**: Restricts to 4 validated safe features:
   - `Contrast` - Measures local intensity variation
   - `Correlation` - Measures linear dependency of gray levels
   - `JointEnergy` - Measures image uniformity (NOT `Energy`!)
   - `Idm` - Inverse Difference Moment, measures local homogeneity
3. **User Control**: If you explicitly list GLCM features, HABIT respects your choice

### Configuration Example

```yaml
# parameter.yaml
featureClass:
  firstorder:     # Enable all first-order features
  glcm:           # Enable GLCM (auto-restricted to safe features)
```

### Important Notes

- Only applies to `voxel_radiomics()` method
- For ROI-level radiomics, full GLCM feature set can be used
- To use more GLCM features, increase `kernelRadius` (increases computation time)
- Avoid deprecated feature names like `Homogeneity1`, `Homogeneity2`
- The safe feature names are validated against PyRadiomics source code

### Technical Details

The implementation in `habit/core/habitat_analysis/extractors/voxel_radiomics_extractor.py`:
- Checks `extractor.enabledFeatures` dictionary after initialization
- If GLCM features list is empty (meaning all features enabled), applies restriction
- Uses `extractor.enableFeaturesByName(glcm=[...])` API for precise control
- No temporary files or YAML modifications needed

## 🔄 Next Steps (Optional)

If needed, we can create annotated templates for:
- `config_habitat_test_retest.yaml`
- `config_image_preprocessing_dcm2nii.yaml`
- Any other specialized configurations

## 📚 Related Documentation

- **Configuration Index**: `config/README_CONFIG.md`
- **Main README**: `README.md` / `README_en.md`
- **Module Docs**: `doc/` and `doc_en/` directories
- **CLI Documentation**: `HABIT_CLI.md`

---

**Date Created**: 2025-10-19
**Total Templates**: 8 major modules
**Total Lines**: ~3000+ lines of detailed annotations
**Language**: English

