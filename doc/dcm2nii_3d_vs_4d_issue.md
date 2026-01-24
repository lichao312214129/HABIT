# dcm2niix 3D vs 4D Output Issue

## 问题描述

在使用Python的`subprocess.run`或`os.system`运行dcm2niix时，可能会遇到以下情况：

- **Python调用**：输出4D图像（例如40层）
- **终端直接运行**：输出3D图像（例如80层）

## 问题原因

dcm2niix在处理包含多个回波（echo）、多个时相（phase）或多序列的DICOM数据时，会根据参数决定如何组织输出：

1. **缺少`-m`参数（merge slices）**：
   - 默认情况下，dcm2niix会尝试合并2D切片
   - 但在某些情况下（特别是多回波数据），可能会分割成多个3D卷或4D卷

2. **缺少`-s`参数（single file mode）**：
   - 如果没有强制单文件模式，dcm2niix可能会输出多个文件
   - 或者将数据组织为4D格式（时间维度）

## 解决方案

### 1. 添加关键参数

在`habit/core/preprocessing/dcm2niix_converter.py`中添加了两个关键参数：

#### `merge_slices` (对应dcm2niix的`-m`参数)
可选值：
- **`"y"` 或 `"1"`**：默认合并行为（合并2D切片）
- **`"2"`（推荐用于3D输出）**：基于序列合并（更积极的合并策略）
- **`"n"` 或 `"0"`**：不合并，保持原始结构
- **`None`**：不指定参数，使用dcm2niix默认行为

#### `single_file_mode` (对应dcm2niix的`-s`参数)
可选值：
- **`None`（强烈推荐）**：不指定参数，让dcm2niix自动决定最佳输出格式
- **`True`**：强制单文件输出（`-s y`），可能保留4D结构
- **`False`**：允许多文件输出（`-s n`），可能分割卷

### 2. 配置文件设置

在您的配置文件中添加这两个参数：

```yaml
Preprocessing:
  dcm2nii:
    images: [delay2, delay3, delay5]
    dcm2niix_path: path/to/dcm2niix.exe
    compress: true
    anonymize: false
    merge_slices: "2"        # 使用"2"进行更积极的合并
    single_file_mode: null   # 使用null让dcm2niix自动决定（推荐）
```

**重要提示**：
- 对于4D→3D问题，**推荐设置**：`merge_slices: "2"` 和 `single_file_mode: null`
- `single_file_mode: null` 意味着不传递`-s`参数给dcm2niix
- 如果仍有问题，可尝试将`merge_slices`也设为`null`，完全使用dcm2niix默认行为

### 3. 验证输出

运行转换后，可以使用以下方法验证输出：

#### 使用Python检查图像维度

```python
import SimpleITK as sitk

# 读取转换后的图像
image = sitk.ReadImage('output.nii.gz')

# 获取图像尺寸
size = image.GetSize()
print(f"Image size: {size}")
print(f"Dimensions: {len(size)}D")

# 3D图像应该是 (width, height, depth)
# 例如: (512, 512, 80)

# 4D图像会是 (width, height, depth, time)
# 例如: (512, 512, 40, 2)
```

#### 使用命令行工具

```bash
# 使用ITK-SNAP或3D Slicer打开查看
# 或使用Python脚本
python -c "import SimpleITK as sitk; img=sitk.ReadImage('output.nii.gz'); print(f'Size: {img.GetSize()}, Dimensions: {len(img.GetSize())}D')"
```

## 技术细节

### dcm2niix参数说明

- **`-m y`**: Merge 2D slices from same series (合并同一序列的2D切片)
- **`-m n`**: Do not merge 2D slices (不合并2D切片)
- **`-m 2`**: Merge 2D slices based on series only (仅基于序列合并)

- **`-s y`**: Single file mode (单文件模式)
- **`-s n`**: Multiple file mode (多文件模式，默认)

### 为什么终端和Python结果不同？

可能的原因：
1. **环境变量差异**：终端和Python环境中的PATH或其他环境变量不同
2. **默认参数**：终端可能使用了不同的默认参数或配置文件
3. **工作目录**：不同的工作目录可能影响dcm2niix的行为
4. **dcm2niix版本**：不同版本的dcm2niix可能有不同的默认行为

## 常见问题

### Q: 我已经设置了参数，但仍然得到4D图像怎么办？

A: 按以下步骤排查：

1. **查看实际执行的命令**：
   - 运行Python程序时，控制台会打印实际执行的dcm2niix命令
   - 复制这个命令，直接在终端运行，查看是否有相同问题
   
2. **比较终端命令和Python命令**：
   ```bash
   # 假设Python输出的命令是：
   dcm2niix.exe -b n -l y -m 2 -p n -v y -z y -o output input
   
   # 直接在终端尝试不同参数组合：
   # 选项1: 不使用 -m 和 -s 参数
   dcm2niix.exe -b n -l y -p n -v y -z y -o output input
   
   # 选项2: 使用 -m n (不合并，看是否会输出多个3D文件)
   dcm2niix.exe -b n -l y -m n -p n -v y -z y -o output input
   
   # 选项3: 使用 -m y (默认合并)
   dcm2niix.exe -b n -l y -m y -p n -v y -z y -o output input
   ```

3. **尝试不同的配置组合**：
   ```yaml
   # 选项1: 完全不指定参数（推荐首先尝试）
   merge_slices: null
   single_file_mode: null
   
   # 选项2: 使用 -m 2
   merge_slices: "2"
   single_file_mode: null
   
   # 选项3: 不合并（可能输出多个3D文件）
   merge_slices: "n"
   single_file_mode: null
   ```

4. **检查DICOM数据**：
   - 确认DICOM数据本身是否包含多个时相/回波
   - 查看dcm2niix的输出日志，了解它如何解析DICOM数据
   - 使用`verbose: true`参数获取更多信息

### Q: 如何在不修改代码的情况下临时测试？

A: 可以在终端直接运行dcm2niix命令进行测试：

```bash
dcm2niix.exe -m y -s y -z y -b n -o output_dir input_dicom_dir
```

参数说明：
- `-m y`: 合并切片
- `-s y`: 单文件模式
- `-z y`: 压缩输出
- `-b n`: 不生成JSON文件
- `-o output_dir`: 输出目录
- `input_dicom_dir`: 输入DICOM目录

### Q: 是否应该总是使用merge_slices和single_file_mode？

A: **推荐设置为True**，除非您有特殊需求：
- 如果需要保留多回波或多时相数据的4D结构，可设置为False
- 如果需要分别处理每个时相，可设置single_file_mode为False

## 参考资料

- [dcm2niix GitHub](https://github.com/rordenlab/dcm2niix)
- [dcm2niix Documentation](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage)

## 更新日志

- **2025-10-29**: 添加`merge_slices`和`single_file_mode`参数以解决3D/4D输出问题

