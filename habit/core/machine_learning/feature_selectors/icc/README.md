# 多进程ICC分析工具

这是一个用于计算多个CSV或Excel文件对间组内相关系数(ICC)的多进程分析工具。该工具专为放射组学特征分析设计，可以高效地分析多对数据文件中的ICC值。

## 功能特点

- 支持CSV和Excel格式(.xlsx, .xls)文件
- 多进程并行计算，显著提高处理速度
- 自动处理患者ID不一致的情况，计算共同患者的交集
- 支持按文件对或目录对的方式批量分析
- 详细的日志记录和进度显示
- 日志文件保存在与输出文件相同的目录下
- 结果以JSON格式保存，方便后续处理
- 支持habitat标签映射功能，用于测试-重测分析

## 依赖库

- pandas
- numpy
- pingouin
- multiprocessing
- argparse
- SimpleITK (用于habitat标签映射)
- pyyaml (用于habitat标签映射)

## 安装依赖

```bash
pip install pandas numpy pingouin openpyxl SimpleITK pyyaml
```

## 使用流程

1. 首先使用Habitat标签映射工具对重测数据进行标签重映射
2. 使用重映射后的数据提取特征
3. 最后使用ICC分析工具计算测试-重测的可靠性

## 使用方法

### 1. Habitat标签映射（用于测试-重测分析）

```bash
python habitat_test_retest_mapper.py \
    --test-habitat-table ../data/results/habitats.csv \
    --retest-habitat-table ../data/results_of_icc/habitats.csv \
    --input-dir ../data/results_of_icc \
    --out-dir ../data/results_of_icc \
    --processes 8 \
    --similarity-method pearson \
    --debug
```

### 2. ICC分析

#### 按文件对分析

```bash
python icc.py --file-pairs "path/to/file1.csv,path/to/file2.xlsx;path/to/file3.xlsx,path/to/file4.csv" --output results.json
```

#### 按目录对分析（会匹配目录中同名的数据文件）

```bash
python icc.py --dir-pairs "path/to/dir1,path/to/dir2;path/to/dir3,path/to/dir4" --output results.json
```

#### 指定进程数

```bash
python icc.py --file-pairs "path/to/file1.csv,path/to/file2.xlsx" --processes 4 --output results.json
```

## 参数说明

### Habitat标签映射参数
- `--test-habitat-table`: 测试组的habitat特征表格文件路径（CSV/Excel）
- `--retest-habitat-table`: 重测组的habitat特征表格文件路径（CSV/Excel）
- `--features`: 用于计算相似性的特征名称列表，如果不指定则使用第4到倒数第1列
- `--similarity-method`: 相似度计算方法，可选值：
  - `pearson`: Pearson相关系数（默认）
  - `spearman`: Spearman等级相关
  - `kendall`: Kendall等级相关
  - `euclidean`: 欧氏距离（归一化后取负）
  - `cosine`: 余弦相似度
  - `manhattan`: 曼哈顿距离（归一化后取负）
  - `chebyshev`: 切比雪夫距离（归一化后取负）
- `--input-dir`: 输入NRRD文件目录
- `--out-dir`: 输出目录
- `--processes`: 进程数，默认为2
- `--debug`: 启用调试日志

### ICC分析参数
- `--file-pairs`: 文件对，格式为 "file1.csv,file2.xlsx;file3.xlsx,file4.csv"（使用分号分隔不同的文件对，使用逗号分隔同一文件对中的文件）
- `--dir-pairs`: 目录对，格式为 "dir1,dir2;dir3,dir4"，将匹配目录中同名的数据文件
- `--processes`: 进程数，默认使用所有可用CPU
- `--output`: 输出结果的JSON文件路径，默认为 "icc_results.json"

## 注意事项

1. 使用顺序：先进行Habitat标签映射，再提取特征，最后进行ICC分析
2. CSV/Excel文件的第一列应当是患者ID（作为索引）
3. 工具会自动处理两个文件之间患者ID不一致的情况，只分析共同存在的患者
4. 多进程处理可能会使CPU使用率较高，请根据系统情况适当调整进程数
5. 使用分号`;`分隔不同的文件对，使用逗号`,`分隔同一文件对中的文件
6. 目录对分析会匹配文件名（不含扩展名）相同的文件，即使它们的格式不同（例如dir1/data.csv和dir2/data.xlsx）
7. 标签映射会保持原始图像的几何信息和元数据
8. 相似度计算方法的选择会影响标签映射的结果，建议根据数据特点选择合适的计算方法

## 输出格式

### Habitat标签映射输出
- 重映射后的NRRD文件（文件名后缀为`_remapped.nrrd`）
- 详细的处理日志
- 成功/失败统计信息

### ICC分析输出
输出的JSON文件格式如下：

```json
{
    "file1_vs_file2": {
        "feature1": 0.85,
        "feature2": 0.76,
        ...
    },
    "file3_vs_file4": {
        "feature1": 0.92,
        "feature2": 0.81,
        ...
    }
}
``` 