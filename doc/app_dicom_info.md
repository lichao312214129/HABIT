# DICOM信息提取工具

## 概述

本工具提供了从DICOM文件中提取和查看信息的命令行功能。支持批量读取DICOM文件，可以提取指定的DICOM标签信息，并保存为CSV、Excel或JSON格式。

**主要特性**：
- 支持批量读取DICOM文件（支持目录、单个文件或YAML配置文件）
- 自动按序列分组，避免重复读取同一序列的多个切片文件
- 支持提取指定的DICOM标签或所有标准标签
- 支持多种输出格式（CSV、Excel、JSON）
- 可以列出DICOM文件中可用的所有标签

## 安装要求

### 依赖项
- `pydicom`: DICOM文件读取库
- `pandas`: 数据处理
- `openpyxl`: Excel文件支持（如果使用Excel格式）

### 安装依赖
```bash
pip install pydicom pandas openpyxl
```

或者安装完整的HABIT依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 基本命令格式

```bash
habit dicom-info --input <输入路径> [选项]
```

### 常用选项

| 选项 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | DICOM目录、文件或YAML配置文件路径（必需） | - |
| `--tags` | `-t` | 要提取的DICOM标签，逗号分隔 | 所有标准标签 |
| `--output` | `-o` | 输出文件路径 | 不保存 |
| `--format` | `-f` | 输出格式：csv、excel、json | csv |
| `--recursive` | - | 递归搜索子目录 | 启用 |
| `--no-recursive` | - | 不递归搜索子目录 | - |
| `--group-by-series` | - | 按序列分组，每个序列只读取一个文件 | 启用 |
| `--no-group-by-series` | - | 读取所有文件（不分组） | - |
| `--one-file-per-folder` | - | 每个文件夹只读取一个DICOM文件，大幅加速扫描速度 | 禁用 |
| `--dicom-extensions` | - | DICOM文件扩展名列表，逗号分隔（如 `.dcm,.dicom,.ima`）。仅与 `--one-file-per-folder` 配合使用 | .dcm,.dicom |
| `--include-no-extension` | - | 同时检查无扩展名的文件（通过读取DICOM魔数字节验证）。仅与 `--one-file-per-folder` 配合使用。适用于部分不生成扩展名的医疗设备 | 禁用 |
| `--num-workers` | `-j` | 并行处理的工作线程数。默认：min(32, cpu_count + 4)。设为1禁用并行 | 自动 |
| `--max-depth` | `-d` | 目标文件夹的深度。快速定位到指定深度的文件夹，然后在每个文件夹中找**一个**DICOM作为代表（找到即停止，不遍历所有子文件夹）。0=根目录，1=根+1层。例如：对于 patient/study/series/*.dcm 结构，使用 `-d 2` 在 study 级别找代表文件 | 无限制 |
| `--list-tags` | - | 列出可用标签而不是提取信息 | - |
| `--num-samples` | - | 列出标签时采样的文件数 | 1 |

## 使用示例

### 1. 列出可用的DICOM标签

查看DICOM文件中包含哪些标签：

```bash
habit dicom-info -i /path/to/dicom --list-tags
```

保存标签列表到文件：

```bash
habit dicom-info -i /path/to/dicom --list-tags -o available_tags.txt
```

### 2. 提取指定标签并保存为CSV

提取患者姓名、研究日期和模态信息：

```bash
habit dicom-info -i /path/to/dicom --tags "PatientName,StudyDate,Modality" --output results.csv
```

### 3. 提取所有标准标签并保存为Excel

```bash
habit dicom-info -i /path/to/dicom --output dicom_info.xlsx --format excel
```

### 4. 从YAML配置文件读取

如果您的数据组织在YAML配置文件中：

```yaml
images:
  subject_001:
    T1: /path/to/subject_001/T1_dicom
    T2: /path/to/subject_001/T2_dicom
  subject_002:
    T1: /path/to/subject_002/T1_dicom
```

可以这样提取信息：

```bash
habit dicom-info -i config.yaml --tags "PatientID,SeriesDescription" --output dicom_info.csv
```

### 5. 提取信息但不保存（仅显示）

```bash
habit dicom-info -i /path/to/dicom --tags "PatientID,SeriesNumber,Modality"
```

### 6. 读取所有文件（不按序列分组）

默认情况下，工具会按`SeriesInstanceUID`分组，每个序列只读取一个代表性文件。如果您需要读取所有文件：

```bash
habit dicom-info -i /path/to/dicom --no-group-by-series --output all_files.csv
```

### 7. 非递归搜索

只在指定目录中搜索，不搜索子目录：

```bash
habit dicom-info -i /path/to/dicom --no-recursive --output results.csv
```

## 标准DICOM标签

如果不指定`--tags`选项，工具会提取以下标准标签：

**患者信息**：
- PatientID, PatientName, PatientBirthDate, PatientSex, PatientAge

**研究信息**：
- StudyInstanceUID, StudyDate, StudyTime, StudyDescription

**序列信息**：
- SeriesInstanceUID, SeriesNumber, SeriesDescription, SeriesDate, SeriesTime

**设备信息**：
- Modality, Manufacturer, ManufacturerModelName

**图像参数**：
- SliceThickness, SpacingBetweenSlices, PixelSpacing
- Rows, Columns, BitsAllocated, BitsStored, HighBit
- ImagePositionPatient, ImageOrientationPatient

**扫描参数**：
- EchoTime, RepetitionTime, FlipAngle
- InstanceNumber, SliceLocation
- AcquisitionDate, AcquisitionTime

**对比剂信息**：
- ContrastBolusAgent, ContrastBolusVolume

**X射线参数**（CT）：
- KVP, XRayTubeCurrent, ExposureTime

**窗宽窗位**：
- WindowCenter, WindowWidth

**像素值转换**：
- RescaleIntercept, RescaleSlope

## 序列分组说明

**重要**：默认情况下，工具会按`SeriesInstanceUID`对文件进行分组。这是因为：

1. **一个序列包含多个切片**：在DICOM中，一个序列（series）通常包含多个切片文件（.dcm），每个切片是3D体积的一个层面。

2. **避免重复信息**：同一序列的所有切片文件包含相同的序列级别信息（如PatientID、StudyDate、SeriesDescription等），只有切片特定的信息（如InstanceNumber、SliceLocation）不同。

3. **提高效率**：通过分组，工具只读取每个序列的一个代表性文件，大大减少了处理时间和输出文件大小。

**分组行为**：
- 工具会读取每个序列的第一个文件
- 输出结果中会包含`Files_In_Series`列，显示该序列包含多少个文件
- 如果文件没有`SeriesInstanceUID`，则每个文件单独处理

**禁用分组**：
如果需要查看每个切片文件的详细信息（例如InstanceNumber、SliceLocation），可以使用`--no-group-by-series`选项。

## 输出格式说明

### CSV格式
- 逗号分隔值
- 可以用Excel、文本编辑器或pandas读取
- 推荐用于数据分析和处理

### Excel格式
- .xlsx格式
- 可以用Excel直接打开
- 适合查看和手动编辑

### JSON格式
- 结构化数据格式
- 适合程序化处理
- 每个DICOM文件的信息作为一个JSON对象

## 常见问题

### Q: 为什么输出文件中有`Files_In_Series`列？
A: 当使用`--group-by-series`选项（默认）时，工具会显示每个序列包含多少个文件。这有助于了解数据规模。

### Q: 如何提取自定义标签？
A: 使用`--tags`选项指定标签名称，例如：
```bash
habit dicom-info -i /path/to/dicom --tags "CustomTag1,CustomTag2" --output results.csv
```

### Q: 如果DICOM文件没有SeriesInstanceUID怎么办？
A: 工具会自动处理这种情况。没有SeriesInstanceUID的文件会被单独处理，不会与其他文件分组。

### Q: 可以同时处理多个目录吗？
A: 可以。使用`--recursive`选项（默认启用）可以递归搜索所有子目录。或者使用YAML配置文件指定多个路径。

### Q: 输出文件很大怎么办？
A: 如果输出文件很大，可能是因为：
1. 没有使用序列分组（使用`--group-by-series`）
2. 提取了太多标签（只提取需要的标签）
3. 数据量确实很大（考虑分批处理）

## 技术细节

### 文件识别
工具会自动识别以下扩展名的文件：
- `.dcm`
- `.dicom`

### 序列分组算法
1. 快速读取每个文件的`SeriesInstanceUID`（不读取完整文件）
2. 按`SeriesInstanceUID`分组
3. 每个序列选择第一个文件进行完整读取
4. 没有`SeriesInstanceUID`的文件单独处理

### 错误处理
- 无法读取的文件会被跳过，并在日志中记录警告
- 如果所有文件都无法读取，工具会退出并显示错误信息
- 部分标签缺失不会导致整个文件读取失败

## 相关文档

- [图像预处理文档](app_image_preprocessing.md)
- [DICOM转换文档](app_dcm2nii.md)

