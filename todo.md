docs
1 机器学习的配置是否有问题。少不少文件。对照demo data

2 docs中也要提供demo压缩数据下载链接（百度网盘）

3 快速开始的下一步链接的第一个是不是有问题

4 docs中部分api没有使用Service模式。要不要把包中的相关部分统一用服务模式？

5 Direct Pooling描述有问题，优化。比如应该是所有人所有体素一次性拼接，也不一定就泄露信息
6 建议在有训练和预测之分的场景，尽量把两种方式的使用说明都展示。
7 生境分析模块，特征的处理的目的和意义，分个体和群体。
8 生境分析的文档，常见问题部分，有些格式不对，换行有问题。也检查文档其它部分，修复。
9 文档中的安装和readme中安装不一致。先要新建环境。readme中安装比较合适。

10 [代码优化] ✅ 已完成 - 将包中相关API统一重构为Service模式
   - ✅ 创建 RadiomicsConfig 和 TestRetestConfig 配置类
   - ✅ 重构 cmd_radiomics.py 和 cmd_test_retest.py 使用 Service 模式
   - ✅ 在 ServiceConfigurator 中添加 create_radiomics_extractor() 和 create_test_retest_analyzer()
   - ✅ 更新 COMMANDS_MIGRATION_STATUS.md 反映迁移状态
   - ✅ 更新所有文档示例为 Service 模式（preprocessing、feature_extraction、machine_learning）

11 [代码优化] ✅ 已完成 - 优化Direct Pooling在代码中的实现逻辑与注释说明
   - ✅ 在 direct_pooling_strategy.py 中添加详细注释
   - ✅ 说明为什么拼接所有体素（发现群体共性组织模式）
   - ✅ 澄清关于信息泄露的误解（无监督学习、特征空间操作）
   - ✅ 说明适用场景（快速原型、群体模式发现）

12 [代码优化] ✅ 已完成 - 在生境分析模块代码中补充特征处理（个体/群体）的逻辑注释
   - ✅ 在 feature_manager.py 的 _apply_preprocessing() 方法添加详细注释
   - ✅ Subject-level: 消除个体内异常值和量级差异，确保可比性
   - ✅ Group-level: 通过离散化减少噪声，捕捉稳定的生物学模式

13 [发布方案] 实现 pip install 和 conda install 安装方式
    13.1 [PyPI发布] 完善 setup.py/pyproject.toml，添加 long_description、url、project_urls、keywords 等字段
    13.2 [PyPI发布] 创建 MANIFEST.in 文件，指定需要包含的非Python文件（配置文件模板、数据文件等）
    13.3 [PyPI发布] 统一版本管理：在 setup.py 和 habit/__init__.py 中统一版本号，使用语义化版本（如 1.0.0）
    13.4 [PyPI发布] 创建 GitHub Actions workflow，实现自动发布到 PyPI（检测到新 tag 时自动构建和发布）
    13.5 [PyPI发布] 测试发布到 TestPyPI，验证安装和功能正常
    13.6 [PyPI发布] 正式发布到 PyPI，确保包名可用（建议使用小写 "habit"）
    13.7 [Conda发布] 创建 conda recipe（meta.yaml、build.sh、bld.bat），支持多平台构建
    13.8 [Conda发布] 测试本地 conda 构建，验证依赖和功能
    13.9 [Conda发布] 提交到 conda-forge 或自建 conda channel（conda-forge 需要审核，自建 channel 需要维护服务器）
    13.10 [文档更新] 更新安装文档，添加 pip install habit 和 conda install habit 的安装说明
    13.11 [文档更新] 在 README.md 中添加 PyPI 和 Conda 安装方式的徽章和说明
