# HABIT 命令行界面使用文档

> 📌 **注意**: 本文档已整合到主 CLI 文档中

## 📖 查看完整文档

请查看项目根目录的 [**HABIT_CLI.md**](../HABIT_CLI.md) 文件，获取：

- ⚡ 快速开始指南
- 🔧 详细安装步骤
- 📋 所有命令说明
- 💡 完整使用示例
- ❓ 常见问题解答

## 🚀 快速命令参考

### 基本使用

```bash
# 使用 Python 模块方式（推荐）
python -m habit --help
python -m habit <命令> -c <配置文件>
```

### 所有命令

| 命令 | 说明 |
|------|------|
| `preprocess` | 图像预处理 |
| `habitat` | 生成 Habitat 地图 |
| `extract-features` | 提取特征 |
| `ml` | 机器学习（训练/预测） |
| `kfold` | K折交叉验证 |
| `compare` | 模型比较 |
| `icc` | ICC 分析 |
| `radiomics` | 传统影像组学 |
| `test-retest` | Test-Retest 分析 |

### 示例

```bash
# 图像预处理
python -m habit preprocess -c config/config_image_preprocessing.yaml

# 训练模型
python -m habit ml -c config/config_machine_learning.yaml -m train

# 预测
python -m habit ml -c config/config_machine_learning.yaml \
  -m predict \
  --model ./model.pkl \
  --data ./data.csv \
  -o ./output/
```

---

📚 **完整文档**: [HABIT_CLI.md](../HABIT_CLI.md)
