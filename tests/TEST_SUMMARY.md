# HABIT 测试套件总结

## 📊 测试文件总览

### 核心测试文件
| 文件 | 用途 | 状态 |
|------|------|------|
| `conftest.py` | Pytest fixtures和配置 | ✅ |
| `pytest.ini` | Pytest配置文件 | ✅ |

### Debug脚本（9个）
| 文件 | 测试模块 | 配置文件 |
|------|---------|---------|
| `debug_preprocess.py` | 图像预处理 | `config_image_preprocessing.yaml` |
| `debug_habitat.py` | 栖息地分析 | `config_habitat_analysis.yaml` |
| `debug_extract_features.py` | 特征提取 | `config_feature_extraction.yaml` |
| `debug_radiomics.py` | 影像组学 | `config_radiomics.yaml` |
| `debug_ml.py` | 机器学习 | `config_ml.yaml` |
| `debug_kfold.py` | K折交叉验证 | `config_kfold.yaml` |
| `debug_icc.py` | ICC分析 | `config_icc.yaml` |
| `debug_test_retest.py` | 重测信度 | `config_test_retest.yaml` |
| `debug_compare.py` | 模型比较 | `config_compare.yaml` |

### 单元测试文件（5个）
| 文件 | 测试内容 | 测试类数量 |
|------|---------|-----------|
| `test_preprocessing.py` | 预处理模块 | 5 |
| `test_habitat_analysis.py` | 栖息地分析 | 3 |
| `test_machine_learning.py` | 机器学习 | 7 |
| `test_utils.py` | 工具函数 | 7 |
| `test_cli.py` | CLI命令 | 2 |

### 测试运行器（2个）
| 文件 | 功能 |
|------|------|
| `run_all_tests.py` | 运行所有单元测试并生成覆盖率报告 |
| `run_debug_menu.py` | 交互式调试菜单 |

### 文档（5个）
| 文件 | 内容 |
|------|------|
| `README.md` | 测试文档（英文） |
| `TESTING_GUIDE.md` | 详细测试指南（中文） |
| `QUICKSTART.md` | 5分钟快速开始 |
| `TEST_CHECKLIST.md` | 测试覆盖清单 |
| `TEST_SUMMARY.md` | 本文件 |

---

## 🎯 快速使用

### 场景1：快速调试单个模块
```bash
# 使用交互式菜单
python tests/run_debug_menu.py

# 或直接运行
python tests/debug_preprocess.py
```

### 场景2：运行所有单元测试
```bash
# 使用测试运行器
python tests/run_all_tests.py

# 或使用pytest
pytest tests/ -v
```

### 场景3：测试特定功能
```bash
# 测试预处理
pytest tests/test_preprocessing.py -v

# 测试CLI
pytest tests/test_cli.py::TestCLICommands::test_cli_help -v
```

### 场景4：生成覆盖率报告
```bash
pytest tests/ --cov=habit --cov-report=html
# 查看 htmlcov/index.html
```

---

## 📂 项目结构

```
habit_project/
├── habit/                          # 主代码包
│   ├── cli.py                     # CLI入口
│   ├── cli_commands/              # CLI命令
│   ├── core/                      # 核心模块
│   │   ├── preprocessing/         # 预处理
│   │   ├── habitat_analysis/      # 栖息地分析
│   │   └── machine_learning/      # 机器学习
│   └── utils/                     # 工具函数
│
├── tests/                         # 测试套件 ⭐
│   ├── __init__.py
│   ├── conftest.py               # 共享fixtures
│   ├── pytest.ini                # Pytest配置
│   │
│   ├── debug_*.py                # Debug脚本（9个）
│   ├── test_*.py                 # 单元测试（5个）
│   ├── run_*.py                  # 测试运行器（2个）
│   │
│   ├── README.md                 # 测试文档
│   ├── TESTING_GUIDE.md          # 详细指南
│   ├── QUICKSTART.md             # 快速开始
│   ├── TEST_CHECKLIST.md         # 测试清单
│   └── TEST_SUMMARY.md           # 本文件
│
├── demo_image_data/              # 测试数据和配置
│   ├── config_*.yaml             # 配置文件（9个）
│   └── ...
│
├── pyproject.toml                # 项目配置
└── README.md                     # 项目文档
```

---

## 🔧 测试框架

### 依赖
```
pytest>=6.2.5           # 测试框架
pytest-cov              # 覆盖率报告（可选）
pytest-xdist            # 并行测试（可选）
```

### 安装
```bash
pip install pytest pytest-cov pytest-xdist
```

---

## 📝 测试命令速查表

| 命令 | 功能 |
|------|------|
| `pytest tests/` | 运行所有测试 |
| `pytest tests/ -v` | 详细输出 |
| `pytest tests/ -s` | 显示print输出 |
| `pytest tests/ -x` | 首次失败即停止 |
| `pytest tests/ --pdb` | 失败时进入调试器 |
| `pytest tests/ -k "test_cli"` | 运行名称匹配的测试 |
| `pytest tests/ -m "unit"` | 运行特定标记的测试 |
| `pytest tests/ --lf` | 只运行上次失败的 |
| `pytest tests/ --cov=habit` | 生成覆盖率报告 |
| `pytest tests/ -n auto` | 并行运行（需要xdist） |

---

## 📈 测试统计

### 当前状态
- **Debug脚本**: 9个 ✅
- **单元测试文件**: 5个 ✅
- **测试类**: 24个 ✅
- **测试函数**: 约80个（框架已创建，待实现）
- **代码覆盖率**: 待测量

### 测试分类
```
tests/
├── Debug Scripts (9)        # 集成测试/调试
├── Unit Tests (5)          # 单元测试
│   ├── test_preprocessing.py      (5 classes)
│   ├── test_habitat_analysis.py   (3 classes)
│   ├── test_machine_learning.py   (7 classes)
│   ├── test_utils.py              (7 classes)
│   └── test_cli.py                (2 classes)
└── Test Runners (2)        # 测试工具
```

---

## 🎓 学习路径

### 初学者
1. 阅读 `QUICKSTART.md`（5分钟快速上手）
2. 运行 `test_cli.py` 中的简单测试
3. 使用 `run_debug_menu.py` 尝试不同模块

### 中级用户
1. 阅读 `TESTING_GUIDE.md`（详细指南）
2. 编写自己的单元测试
3. 运行覆盖率分析

### 高级用户
1. 查看 `TEST_CHECKLIST.md`（测试清单）
2. 实现缺失的测试用例
3. 优化测试性能和覆盖率

---

## 🚀 下一步计划

### 短期（1-2周）
- [ ] 实现 P0 级别测试（CLI、配置、图像加载）
- [ ] 完成预处理模块的核心测试
- [ ] 设置 CI/CD 自动测试

### 中期（1-2月）
- [ ] 完成所有单元测试实现
- [ ] 达到 80% 代码覆盖率
- [ ] 添加集成测试

### 长期（3-6月）
- [ ] 添加性能基准测试
- [ ] 实现压力测试
- [ ] 完善文档和示例

---

## 💡 最佳实践

1. **每次修改代码后运行相关测试**
   ```bash
   pytest tests/test_preprocessing.py -v
   ```

2. **提交前运行所有测试**
   ```bash
   python tests/run_all_tests.py
   ```

3. **定期检查覆盖率**
   ```bash
   pytest tests/ --cov=habit --cov-report=term-missing
   ```

4. **使用debug脚本快速验证功能**
   ```bash
   python tests/debug_preprocess.py
   ```

5. **遵循测试命名规范**
   - 文件名：`test_*.py`
   - 类名：`Test*`
   - 函数名：`test_*`

---

## 🐛 故障排查

### 问题：找不到habit模块
```bash
# 解决：安装项目为可编辑模式
pip install -e .
```

### 问题：配置文件路径错误
```python
# 解决：在debug脚本中使用绝对路径
sys.argv = ['habit', 'preprocess', '-c', 'F:/path/to/config.yaml']
```

### 问题：pytest找不到
```bash
# 解决：安装pytest
pip install pytest
```

### 问题：测试失败
```bash
# 查看详细错误
pytest tests/ -v --tb=long

# 进入调试模式
pytest tests/ --pdb
```

---

## 📞 支持与贡献

- **查看完整指南**: `TESTING_GUIDE.md`
- **快速开始**: `QUICKSTART.md`
- **测试清单**: `TEST_CHECKLIST.md`
- **项目文档**: 根目录 `README.md`

---

## 📅 更新记录

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2025-11-10 | v1.0 | 初始测试套件创建 |

---

**Happy Testing! 🎉**

