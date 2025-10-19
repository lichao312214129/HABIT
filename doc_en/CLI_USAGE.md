# HABIT Command-Line Interface Documentation

> 📌 **Note**: This document has been consolidated into the main CLI guide

## 📖 View Complete Documentation

Please see [**HABIT_CLI.md**](../HABIT_CLI.md) in the project root for:

- ⚡ Quick start guide
- 🔧 Detailed installation steps
- 📋 All commands reference
- 💡 Complete usage examples
- ❓ FAQ and troubleshooting

## 🚀 Quick Command Reference

### Basic Usage

```bash
# Use Python module method (recommended)
python -m habit --help
python -m habit <command> -c <config-file>
```

### All Commands

| Command | Description |
|---------|-------------|
| `preprocess` | Image preprocessing |
| `habitat` | Generate Habitat maps |
| `extract-features` | Extract features |
| `ml` | Machine learning (train/predict) |
| `kfold` | K-fold cross-validation |
| `compare` | Model comparison |
| `icc` | ICC analysis |
| `radiomics` | Traditional radiomics |
| `test-retest` | Test-retest analysis |

### Examples

```bash
# Image preprocessing
python -m habit preprocess -c config/config_image_preprocessing.yaml

# Train model
python -m habit ml -c config/config_machine_learning.yaml -m train

# Predict
python -m habit ml -c config/config_machine_learning.yaml \
  -m predict \
  --model ./model.pkl \
  --data ./data.csv \
  -o ./output/
```

---

📚 **Complete Guide**: [HABIT_CLI.md](../HABIT_CLI.md)
