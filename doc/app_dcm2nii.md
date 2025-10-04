# DICOM to NIfTI Conversion Module

## Overview

本模块提供了使用dcm2niix工具将DICOM文件批量转换为NIfTI格式的功能。该模块已集成到HABIT的预处理流水线中，通过`Dcm2niixConverter`类实现，支持单个和批量转换操作。

**注意**: 此功能不是独立的应用脚本，而是作为预处理流水线的一部分使用。如需使用，请参考`app_image_preprocessing.py`。

## Requirements

### Dependencies
- `dcm2niix` executable (必须安装并可在PATH中访问)
- Python packages: `subprocess`, `pathlib`, `logging`
- HABIT dependencies: `BasePreprocessor`, `CustomTqdm`, `file_system_utils`

### Installing dcm2niix
1. 下载dcm2niix: https://github.com/rordenlab/dcm2niix
2. 将dcm2niix.exe添加到系统PATH中
3. 验证安装: `dcm2niix --help`

## Usage

### 1. 单个DICOM目录转换

```python
from habit.core.preprocessing import Dcm2niixConverter

# Initialize converter
converter = Dcm2niixConverter(
    keys=["dicom_path"],
    output_dir="/path/to/output",
    dcm2niix_executable="dcm2niix.exe",  # Windows系统
    compress=True,
    anonymize=True
)

# Setup data
data = {
    "subject_id": "subject_001",
    "dicom_path": "/path/to/dicom/directory"
}

# Convert
result = converter(data)
```

### 2. 批量转换多个被试

```python
from habit.core.preprocessing import batch_convert_dicom_directories

# Setup data structure
subjects_data = {
    "subject_001": {
        "T1": "/path/to/subject_001/T1_dicom",
        "T2": "/path/to/subject_001/T2_dicom"
    },
    "subject_002": {
        "T1": "/path/to/subject_002/T1_dicom"
    }
}

# Batch convert
converted_files = batch_convert_dicom_directories(
    input_mapping=subjects_data,
    output_dir="/path/to/output",
    dcm2niix_executable="dcm2niix.exe",
    compress=True,
    anonymize=True
)
```

### 3. 集成到预处理流水线

```python
from habit.core.preprocessing import PreprocessorFactory

# Pipeline configuration
pipeline_config = [
    {
        "type": "dcm2niix_converter",
        "params": {
            "keys": ["dicom_path"],
            "output_dir": "/path/to/output",
            "dcm2niix_executable": "dcm2niix.exe",
            "compress": True
        }
    },
    {
        "type": "load_image",
        "params": {
            "keys": ["dicom_path_nifti"]
        }
    }
]

# Create and apply preprocessors
for step in pipeline_config:
    preprocessor = PreprocessorFactory.create(step["type"], **step["params"])
    data = preprocessor(data)
```

## Configuration Parameters

### Dcm2niixConverter Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keys` | Union[str, List[str]] | Required | DICOM目录路径的键名 |
| `output_dir` | str | Required | 输出目录路径 |
| `dcm2niix_executable` | str | "dcm2niix" | dcm2niix可执行文件路径 |
| `filename_format` | Optional[str] | None | 输出文件名格式 |
| `compress` | bool | True | 是否压缩输出文件 |
| `anonymize` | bool | True | 是否匿名化文件名 |
| `ignore_derived` | bool | True | 是否忽略衍生图像 |
| `crop_images` | bool | True | 是否裁剪图像 |
| `verbose` | bool | False | 是否输出详细信息 |
| `batch_mode` | bool | True | 是否启用批处理模式 |
| `allow_missing_keys` | bool | False | 是否允许缺失键 |

### dcm2niix Command Line Options

本模块使用的dcm2niix选项对应关系：

- `-f`: filename_format (输出文件名格式)
- `-i y`: ignore_derived=True (忽略衍生图像)
- `-l y`: batch_mode=True (批处理模式)
- `-p y`: anonymize=True (文件名匿名化)
- `-x y`: crop_images=True (裁剪图像)
- `-v y`: verbose=True (详细输出)
- `-z y`: compress=True (压缩输出)
- `-o`: output_dir (输出目录)

## Output Structure

转换后的文件按以下结构组织：

```
output_dir/
├── subject_001/
│   ├── T1/
│   │   └── subject_001_T1.nii.gz
│   └── T2/
│       └── subject_001_T2.nii.gz
├── subject_002/
│   └── T1/
│       └── subject_002_T1.nii.gz
└── conversion_summary.json
```

## Data Flow

### Input Data Format
```python
data = {
    "subject_id": "subject_001",
    "dicom_path": "/path/to/dicom/directory",
    # ... other keys
}
```

### Output Data Format
```python
data = {
    "subject_id": "subject_001",
    "dicom_path": "/path/to/dicom/directory",
    "dicom_path_nifti": "/path/to/output/subject_001/T1/subject_001_T1.nii.gz",
    "dicom_path_dcm2niix_meta": {
        "original_dicom_dir": "/path/to/dicom/directory",
        "converted_files": {"T1": "/path/to/output/subject_001/T1/subject_001_T1.nii.gz"},
        "conversion_params": {...}
    },
    # ... other keys
}
```

## Error Handling

### Common Errors

1. **dcm2niix not found**
   ```
   RuntimeError: dcm2niix executable not found: dcm2niix.exe
   ```
   - 解决方案: 确保dcm2niix安装并在PATH中

2. **DICOM directory not found**
   ```
   FileNotFoundError: Input directory does not exist: /path/to/dicom
   ```
   - 解决方案: 检查DICOM目录路径是否正确

3. **No NIfTI files created**
   ```
   RuntimeError: No NIfTI files were created for /path/to/dicom
   ```
   - 解决方案: 检查DICOM文件是否有效，dcm2niix参数是否正确

### Error Recovery

- 设置 `allow_missing_keys=True` 可以跳过失败的转换
- 检查日志输出了解详细错误信息
- 使用 `verbose=True` 获取dcm2niix的详细输出

## Examples

### Example 1: 作为预处理流水线的一部分使用
```bash
# 在配置文件中配置dcm2niix转换步骤
# 参考 config/config_image_preprocessing.yaml
python scripts/app_image_preprocessing.py --config ./config/config_image_preprocessing.yaml
```

### Example 2: 自定义参数
```python
converter = Dcm2niixConverter(
    keys=["dicom_path"],
    output_dir="./converted_nifti",
    filename_format="%p_%s",  # 患者ID_序列名
    compress=False,  # 不压缩
    anonymize=False,  # 不匿名化
    verbose=True  # 详细输出
)
```

### Example 3: 多序列转换
```python
subjects_data = {
    "patient_001": {
        "T1_MPRAGE": "/data/patient_001/T1_MPRAGE_dicom",
        "T2_FLAIR": "/data/patient_001/T2_FLAIR_dicom",
        "DWI": "/data/patient_001/DWI_dicom"
    }
}

converted = batch_convert_dicom_directories(
    input_mapping=subjects_data,
    output_dir="./multi_sequence_nifti",
    dcm2niix_executable="dcm2niix.exe"
)
```

## Performance Considerations

1. **Progress Tracking**: 使用CustomTqdm显示转换进度
2. **Memory Usage**: 转换过程不会将图像数据加载到内存中
3. **Disk Space**: 确保输出目录有足够的磁盘空间
4. **Parallel Processing**: 可以通过multiprocessing并行处理多个被试

## Integration with HABIT Pipeline

本模块完全集成到HABIT预处理流水线中：

1. **Factory Registration**: 通过`@PreprocessorFactory.register("dcm2niix_converter")`注册
2. **Base Class Inheritance**: 继承自`BasePreprocessor`
3. **Progress Reporting**: 使用统一的`CustomTqdm`进度条
4. **Error Handling**: 支持`allow_missing_keys`参数
5. **Metadata Management**: 自动生成转换元数据

## Troubleshooting

### 常见问题解决

1. **Windows路径问题**: 使用正斜杠或原始字符串 r"C:\path\to\dir"
2. **权限问题**: 确保对输出目录有写权限
3. **文件名冲突**: 使用唯一的filename_format避免覆盖
4. **DICOM格式问题**: 检查DICOM文件是否符合标准

### 调试建议

1. 设置 `verbose=True` 查看dcm2niix输出
2. 检查日志文件中的错误信息
3. 使用小数据集先测试
4. 验证dcm2niix独立运行是否正常

## Future Enhancements

1. **Parallel Processing**: 支持多进程并行转换
2. **Advanced Filtering**: 根据DICOM标签过滤序列
3. **Quality Control**: 自动检查转换质量
4. **BIDS Compliance**: 支持BIDS格式输出
5. **GUI Interface**: 图形界面支持 