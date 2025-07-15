# CSV Files Merger Scripts

这个目录包含了两个用于合并CSV文件的脚本，可以将多个CSV文件按照第一列（index）进行横向拼接。

## 脚本说明

### 1. `merge_csv_files.py` - 命令行版本
这是一个功能完整的命令行工具，支持通过命令行参数指定所有配置。

**使用方法：**
```bash
python merge_csv_files.py -i "H:/results/features" -n "file1,file2,file3" -o "merged_output.csv"
```

**参数说明：**
- `-i, --input-folder`: 包含CSV文件的文件夹路径
- `-n, --csv-names`: 要合并的CSV文件名列表（用逗号分隔，不包含.csv扩展名）
- `-o, --output-file`: 输出合并后的CSV文件路径
- `--index-col`: 用作索引的列名（可选，默认使用第一列）
- `--separator`: CSV分隔符（可选，默认为逗号）
- `--encoding`: 文件编码（可选，默认为utf-8）

**使用示例：**
```bash
# 基本用法
python merge_csv_files.py -i "H:/results/features" -n "clinical_data,radiomics_features,habitat_features" -o "merged_features.csv"

# 指定索引列
python merge_csv_files.py -i "H:/results/features" -n "file1,file2" -o "merged.csv" --index-col "subject_id"

# 使用分号分隔符
python merge_csv_files.py -i "H:/results/features" -n "file1,file2" -o "merged.csv" --separator ";"
```

### 2. `merge_csv_simple.py` - 简化版本
这是一个简化版本，可以直接在代码中修改配置参数，适合快速使用。

**使用方法：**
1. 打开 `merge_csv_simple.py` 文件
2. 修改配置部分：
   ```python
   # 输入文件夹路径
   INPUT_FOLDER = r"H:\results\features"
   
   # 要合并的CSV文件名列表（不包含.csv扩展名）
   CSV_NAMES = [
       "clinical_data",
       "radiomics_features", 
       "habitat_features",
       # 添加更多文件名
   ]
   
   # 输出文件路径
   OUTPUT_FILE = r"H:\results\features\merged_features.csv"
   
   # 索引列名（可选，None表示使用第一列）
   INDEX_COL = None
   ```
3. 运行脚本：
   ```bash
   python merge_csv_simple.py
   ```

## 功能特点

1. **自动索引处理**: 自动使用第一列作为索引进行合并
2. **列名冲突避免**: 当合并多个文件时，自动为列名添加前缀以避免冲突
3. **进度显示**: 使用项目统一的进度条显示处理进度
4. **错误处理**: 完善的错误处理和日志记录
5. **灵活配置**: 支持自定义分隔符、编码等参数
6. **外连接合并**: 使用外连接确保所有行的数据都被保留

## 合并逻辑

- 脚本会读取每个指定的CSV文件
- 将第一列（或指定的索引列）设置为索引
- 使用pandas的join方法进行横向合并
- 如果合并多个文件，会为列名添加文件名前缀以避免冲突
- 最终保存合并后的文件

## 注意事项

1. **文件格式**: 确保所有CSV文件格式一致，第一列应该是相同的标识符
2. **文件存在性**: 脚本会检查文件是否存在，不存在的文件会被跳过并记录警告
3. **内存使用**: 对于大型文件，请确保有足够的内存
4. **编码问题**: 如果遇到编码问题，可以尝试不同的编码格式（如gbk、utf-8-sig等）

## 输出示例

合并后的CSV文件格式：
```
subject_id,clinical_data_age,clinical_data_gender,radiomics_features_feature1,habitat_features_feature2
001,45,M,0.123,0.456
002,52,F,0.234,0.567
003,38,M,0.345,0.678
```

## 依赖项

- pandas
- habit.utils.progress_utils
- habit.utils.log_utils

确保这些依赖项已正确安装。 