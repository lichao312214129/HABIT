# 配置访问方式迁移总结

## 🎯 问题根源

在将配置从字典迁移到 Pydantic 对象后，部分代码仍在使用 `config.get()` 字典访问方式，导致：
1. 无法利用 Pydantic 的类型安全特性
2. 无法获得 IDE 自动补全
3. 运行时可能出错（如果 config 是 Pydantic 对象但没有 `.get()` 方法）

## ✅ 已修复的文件

### 1. holdout_workflow.py
**修复内容**：
```python
# 修复前
models_config = self.config.get('models', {})

# 修复后
if self.config_obj is not None:
    models_config = {
        name: params.params  # 提取 params 字典
        for name, params in self.config_obj.models.items()
    }
else:
    models_config = self.config.get('models', {})
```

### 2. kfold_workflow.py
**修复内容**：
- K-Fold 配置访问：使用 `self.config_obj.stratified` 和 `self.config_obj.n_splits`
- Models 配置访问：同 holdout_workflow.py

### 3. data_manager.py
**修复内容**：
- 支持 Pydantic 对象和字典两种格式
- 在初始化时提取所有需要的配置值
- 在 `split_data()` 中使用存储的属性

### 4. pipeline_utils.py
**修复内容**：
- `normalization` 配置：支持 Pydantic 对象访问
- `feature_selection_methods`：支持 Pydantic 对象，并转换为字典列表

### 5. visualization_callback.py
**修复内容**：
- 使用 `self.workflow.config_accessor.get()` 统一访问

### 6. model_checkpoint.py
**修复内容**：
- 使用 `self.workflow.config_accessor.get()` 统一访问

### 7. plot_manager.py
**修复内容**：
- 支持 Pydantic 对象和字典两种格式

## 🔑 关键发现

### ModelConfig 对象转换
`MLConfig.models` 的类型是 `Dict[str, ModelConfig]`，其中：
- 键：模型名称（字符串）
- 值：`ModelConfig` 对象（包含 `params` 字段）

**重要**：必须提取 `params` 字段才能得到参数字典：
```python
# 正确方式
models_config = {
    name: params.params  # ModelConfig.params 是 Dict[str, Any]
    for name, params in self.config_obj.models.items()
}
```

## 📋 修复模式

### 模式 1: 直接属性访问（推荐）
```python
if self.config_obj is not None:
    value = self.config_obj.field_name
else:
    value = self.config.get('field_name', default)
```

### 模式 2: 使用 ConfigAccessor（统一访问）
```python
value = self.config_accessor.get('field_name', default)
```

### 模式 3: 支持两种格式（向后兼容）
```python
if hasattr(config, 'field_name'):
    value = config.field_name  # Pydantic object
else:
    value = config.get('field_name', default)  # Dict
```

## ⚠️ 注意事项

1. **ModelConfig 对象**：必须提取 `.params` 字段才能得到参数字典
2. **向后兼容**：所有修复都保持了向后兼容性
3. **类型安全**：优先使用 Pydantic 对象属性访问
4. **Fallback 机制**：当 Pydantic 对象不可用时自动回退

## 🧪 测试建议

1. 使用 MLConfig 对象测试所有 workflow
2. 验证 models 配置正确提取
3. 测试向后兼容性（字典配置）
4. 验证所有配置访问路径正常工作
