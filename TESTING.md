# HABIT 测试指南

本项目包含完整的测试套件，包括Debug脚本和单元测试。

## 📁 测试文件位置

所有测试文件位于 `tests/` 目录下。

## 🚀 快速开始

### 1. 安装测试依赖
```bash
pip install pytest pytest-cov
```

### 2. 快速测试（使用交互式菜单）
```bash
python tests/run_debug_menu.py
```

### 3. 运行所有单元测试
```bash
python tests/run_all_tests.py
```

## 📂 测试文件结构

```
tests/
├── debug_*.py              # Debug脚本 (9个) - 快速调试特定模块
├── test_*.py               # 单元测试 (5个) - 全面的功能测试
├── run_all_tests.py        # 运行所有测试
├── run_debug_menu.py       # 交互式菜单
└── 文档/
    ├── QUICKSTART.md       # 5分钟快速入门 ⭐ 推荐新手阅读
    ├── TESTING_GUIDE.md    # 详细测试指南
    ├── TEST_CHECKLIST.md   # 测试覆盖清单
    └── TEST_SUMMARY.md     # 测试套件总结
```

## 🎯 测试类型

### 1. Debug脚本 - 快速调试
模拟CLI命令，适合快速测试单个模块：

```bash
# 测试预处理
python tests/debug_preprocess.py

# 测试栖息地分析
python tests/debug_habitat.py

# 测试机器学习
python tests/debug_ml.py
```

**注意**: Debug脚本需要对应的配置文件（在`demo_image_data/`目录）

### 2. 单元测试 - 全面测试
使用pytest框架的完整单元测试：

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_preprocessing.py -v

# 运行特定测试类
pytest tests/test_cli.py::TestCLICommands -v

# 生成覆盖率报告
pytest tests/ --cov=habit --cov-report=html
```

## 📚 详细文档

- **新手入门**: 阅读 `tests/QUICKSTART.md` (5分钟快速上手)
- **详细指南**: 阅读 `tests/TESTING_GUIDE.md` (完整测试指南)
- **测试清单**: 查看 `tests/TEST_CHECKLIST.md` (追踪测试进度)
- **测试总结**: 查看 `tests/TEST_SUMMARY.md` (测试套件概览)

## 🛠️ 常用命令

```bash
# ============ 快速测试 ============
python tests/run_debug_menu.py              # 交互式菜单
python tests/run_all_tests.py              # 运行所有测试

# ============ 单元测试 ============
pytest tests/ -v                            # 运行所有测试（详细）
pytest tests/test_cli.py -v                # 测试CLI模块
pytest tests/test_preprocessing.py -v      # 测试预处理模块
pytest tests/ -k "test_cli" -v             # 运行名称匹配的测试

# ============ 覆盖率 ============
pytest tests/ --cov=habit                   # 覆盖率报告（终端）
pytest tests/ --cov=habit --cov-report=html # HTML覆盖率报告

# ============ Debug调试 ============
pytest tests/ --pdb                         # 失败时进入调试器
pytest tests/ -x                            # 首次失败即停止
pytest tests/ --lf                          # 只运行上次失败的测试
```

## 📋 测试清单

### Debug脚本（9个）
- ✅ `debug_preprocess.py` - 图像预处理
- ✅ `debug_habitat.py` - 栖息地分析
- ✅ `debug_extract_features.py` - 特征提取
- ✅ `debug_radiomics.py` - 影像组学
- ✅ `debug_ml.py` - 机器学习
- ✅ `debug_kfold.py` - K折交叉验证
- ✅ `debug_icc.py` - ICC分析
- ✅ `debug_test_retest.py` - 重测信度
- ✅ `debug_compare.py` - 模型比较

### 单元测试（5个）
- ✅ `test_preprocessing.py` - 预处理模块测试
- ✅ `test_habitat_analysis.py` - 栖息地分析测试
- ✅ `test_machine_learning.py` - 机器学习测试
- ✅ `test_utils.py` - 工具函数测试
- ✅ `test_cli.py` - CLI命令测试

## 🎓 学习路径

1. **第一步**: 阅读 `tests/QUICKSTART.md`
2. **第二步**: 运行 `python tests/run_debug_menu.py` 体验交互式测试
3. **第三步**: 运行 `pytest tests/test_cli.py -v` 查看单元测试
4. **第四步**: 阅读 `tests/TESTING_GUIDE.md` 了解详细用法

## ⚙️ 配置要求

### 运行Debug脚本需要：
1. 在 `demo_image_data/` 目录下准备配置文件：
   - `config_image_preprocessing.yaml`
   - `config_habitat_analysis.yaml`
   - `config_ml.yaml`
   - 等等...

2. 根据实际路径修改配置文件中的路径

### 运行单元测试需要：
```bash
pip install pytest pytest-cov
```

## 🐛 问题排查

### 问题：找不到habit模块
```bash
# 解决：安装项目
pip install -e .
```

### 问题：配置文件路径错误
修改debug脚本中的配置文件路径为你的实际路径

### 问题：pytest找不到
```bash
# 解决：安装pytest
pip install pytest
```

## 📞 获取帮助

- 查看 `tests/QUICKSTART.md` - 快速入门
- 查看 `tests/TESTING_GUIDE.md` - 详细指南
- 查看 `tests/TEST_SUMMARY.md` - 测试总结

---

**开始测试之旅！** 🚀

推荐从这里开始：`python tests/run_debug_menu.py`

