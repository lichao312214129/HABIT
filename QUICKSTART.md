# HABIT 快速开始指南

本指南将帮助您在 5 分钟内快速开始使用 HABIT 进行医学影像栖息地分析。

## 🚀 第一步：验证安装

确保 HABIT 已正确安装：

```bash
# 激活环境
conda activate habit

# 验证安装
python -c "import habit; print('HABIT ready to use!')"
```

## 📁 第二步：准备数据

### 数据目录结构
```
your_data/
├── images/
│   ├── patient001/
│   │   ├── pre_contrast.nrrd
│   │   ├── LAP.nrrd
│   │   ├── PVP.nrrd
│   │   └── delay_3min.nrrd
│   └── patient002/
│       └── ...
└── masks/
    ├── patient001/
    │   └── mask.nrrd
    └── patient002/
        └── mask.nrrd
```

### 支持的图像格式
- NIfTI (.nii, .nii.gz)
- NRRD (.nrrd)
- MetaImage (.mha, .mhd)

## ⚙️ 第三步：配置文件

复制并修改配置文件：

```bash
# 复制示例配置
cp config/config_getting_habitat.yaml my_config.yaml

# 编辑配置文件
nano my_config.yaml  # 或使用其他文本编辑器
```

### 关键配置项
```yaml
# 修改数据路径
data_dir: /path/to/your/data
out_dir: /path/to/output

# 特征提取设置
FeatureConstruction:
  voxel_level:
    method: concat(raw(pre_contrast), raw(LAP), raw(PVP))

# 栖息地分割设置
HabitatsSegmention:
  supervoxel:
    n_clusters: 50
  habitat:
    mode: training
    max_clusters: 10
```

## 🔄 第四步：运行分析

### 完整栖息地分析流程
```bash
# 运行栖息地分析
python scripts/app_getting_habitat_map.py --config my_config.yaml

# 或使用GUI选择配置文件
python scripts/app_getting_habitat_map.py
```

### 输出结果
分析完成后，您将在输出目录中找到：
- `habitat_maps/` - 栖息地图像
- `features/` - 提取的特征
- `clustering_results/` - 聚类结果
- `plots/` - 可视化图表

## 📊 第五步：查看结果

### 栖息地可视化
输出目录包含：
- 每个患者的栖息地图像
- 聚类评估曲线
- 特征分布图表

### 特征数据
- `supervoxel_features.csv` - supervoxel级别特征
- `mean_values_of_all_supervoxels_features.csv` - 群体级特征平均值

## 🔧 常见工作流程

### 1. 影像预处理
```bash
# 如果需要预处理原始影像
python scripts/app_image_preprocessing.py --config config/config_image_preprocessing.yaml
```

### 2. 特征提取
```bash
# 提取传统影像组学特征
python scripts/app_traditional_radiomics_extractor.py --config config/config_traditional_radiomics.yaml

# 提取栖息地特征
python scripts/app_extracting_habitat_features.py --config config/config_extract_features.yaml
```

### 3. 机器学习建模
```bash
# 训练预测模型
python scripts/app_of_machine_learning.py --config config/config_machine_learning.yaml

# 模型比较
python scripts/app_model_comparison_plots.py --config config/config_model_comparison.yaml
```

### 4. 统计分析
```bash
# ICC 分析
python scripts/app_icc_analysis.py --config config/config_icc_analysis.yaml
```

## 🎯 实际案例

### 案例1：肝癌栖息地分析
```bash
# 1. 准备多期相CT影像（动脉期、门脉期等）
# 2. 配置特征提取
# 3. 运行栖息地分析
python scripts/app_getting_habitat_map.py --config liver_config.yaml

# 4. 提取栖息地特征用于预测建模
python scripts/app_extracting_habitat_features.py --config habitat_features_config.yaml

# 5. 训练预测模型
python scripts/app_of_machine_learning.py --config ml_config.yaml
```

## 🔍 故障排除

### 常见错误及解决方案

#### 1. 内存不足
```yaml
# 在配置文件中减少并行进程数
processes: 1

# 减少supervoxel数量
supervoxel:
  n_clusters: 25
```

#### 2. 图像尺寸不匹配
```bash
# 运行预处理脚本进行重采样
python scripts/app_image_preprocessing.py --config preprocess_config.yaml
```

#### 3. 特征提取失败
- 检查图像和掩膜是否正确对应
- 确保掩膜为二值图像
- 验证图像格式和路径

### 查看日志
```bash
# 启用详细日志
python scripts/app_getting_habitat_map.py --config my_config.yaml --debug
```

## 📚 下一步

1. **深入学习**: 阅读 [完整文档](README.md)
2. **自定义分析**: 修改配置文件进行个性化分析
3. **批量处理**: 编写脚本处理多个数据集
4. **结果解释**: 分析栖息地生物学意义

## 💡 提示

- 始终备份原始数据
- 使用相同的预处理参数确保结果一致性
- 定期保存中间结果
- 记录分析参数便于重现

## 📞 获取帮助

- 查看 [详细文档](doc/)
- 提交 [Issues](../../issues)
- 查看 [常见问题解答](FAQ.md)

---

🎉 **恭喜！** 您已经掌握了 HABIT 的基本使用方法。现在可以开始您的医学影像栖息地分析之旅了！ 