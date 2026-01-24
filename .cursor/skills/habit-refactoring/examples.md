# Habit Refactoring Examples

## Real Code Examples from the Codebase

### Example 1: cmd_ml.py Migration

**Before (Legacy):**
```python
# habit/cli_commands/commands/cmd_ml.py (old version)
import yaml
from pathlib import Path
from habit.core.machine_learning.workflows import MLWorkflow

def main(config_path):
    """Run machine learning workflow."""
    # ❌ Direct YAML loading
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # ❌ Direct instantiation
    workflow = MLWorkflow(config)
    workflow.run()
```

**After (Refactored):**
```python
# habit/cli_commands/commands/cmd_ml.py (current version)
from habit.core.machine_learning.config_schemas import MLConfig
from habit.core.common.service_configurator import ServiceConfigurator

def main(config_path):
    """Run machine learning workflow."""
    # ✅ Type-safe config loading
    config = MLConfig.from_file(config_path)

    # ✅ Dependency injection
    workflow = ServiceConfigurator.create_ml_workflow(config)
    workflow.run()
```

### Example 2: cmd_compare.py Migration

**Before (Legacy):**
```python
# ❌ Old pattern with yaml.safe_load
import yaml
from habit.core.machine_learning.workflows.comparison_workflow import ModelComparisonWorkflow

def main(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    workflow = ModelComparisonWorkflow(config)
    workflow.run()
```

**After (Refactored):**
```python
# ✅ New standard pattern
from habit.core.machine_learning.config_schemas import ModelComparisonConfig
from habit.core.common.service_configurator import ServiceConfigurator

def main(config_file):
    config = ModelComparisonConfig.from_file(config_file)
    workflow = ServiceConfigurator.create_model_comparison(config)
    workflow.run()
```

### Example 3: Error Handling Migration

**Before (Poor Error Handling):**
```python
# ❌ Print-based error handling
def main(config_file):
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        service = SomeService(config)
        service.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0
```

**After (Proper Error Handling):**
```python
# ✅ Logging-based error handling
def main(config_file):
    try:
        config = ConfigClass.from_file(config_file)
        service = ServiceConfigurator.create_service(config)
        service.run()
    except Exception as e:
        # Logger is injected by ServiceConfigurator
        service.logger.error(f"Command execution failed: {e}")
        raise
```

## Module-Specific Migration Patterns

### Machine Learning Module

**Config Class:** `MLConfig`
**Factory Method:** `ServiceConfigurator.create_ml_workflow()`

**Migration Pattern:**
```python
# For standard ML workflow
config = MLConfig.from_file(config_path)
workflow = ServiceConfigurator.create_ml_workflow(config)

# For k-fold validation
config = MLConfig.from_file(config_path)
workflow = ServiceConfigurator.create_kfold_workflow(config)
```

### Habitat Analysis Module

**Config Class:** `HabitatAnalysisConfig`
**Factory Method:** `ServiceConfigurator.create_habitat_analysis()`

**Migration Pattern:**
```python
config = HabitatAnalysisConfig.from_file(config_file)
service = ServiceConfigurator.create_habitat_analysis(config)
service.run()
```

### Preprocessing Module

**Config Class:** `PreprocessingConfig`
**Factory Method:** `ServiceConfigurator.create_batch_processor()`

**Migration Pattern:**
```python
config = PreprocessingConfig.from_file(config_path)
processor = ServiceConfigurator.create_batch_processor(config)
processor.run()
```

## Configuration Schema Examples

### MLConfig Structure
```yaml
# config.yaml
data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  target_column: "outcome"

model:
  type: "LogisticRegression"
  parameters:
    C: 1.0
    max_iter: 1000

evaluation:
  metrics: ["accuracy", "precision", "recall"]
  cv_folds: 5
```

### Corresponding Config Class Usage
```python
from habit.core.machine_learning.config_schemas import MLConfig

config = MLConfig.from_file("config.yaml")
# config.data.train_path is automatically resolved to absolute path
# config.model.type is validated to be a known model type
# All parameters are type-checked
```

## ServiceConfigurator Implementation Patterns

### Factory Method Pattern
```python
# habit/core/common/service_configurator.py
class ServiceConfigurator:
    @staticmethod
    def create_ml_workflow(config: MLConfig) -> MLWorkflow:
        """Create ML workflow with all dependencies injected."""
        logger = cls._create_logger(config)
        validator = cls._create_validator(config)
        data_manager = cls._create_data_manager(config)

        return MLWorkflow(
            config=config,
            logger=logger,
            validator=validator,
            data_manager=data_manager
        )
```

### Dependency Injection Benefits
- **Testability**: Easy to mock dependencies
- **Consistency**: All services get same logger/validator instances
- **Maintainability**: Dependency changes isolated to configurator
- **Flexibility**: Can swap implementations without changing commands

## Testing Migration Examples

### Before: Hard to Test
```python
# ❌ Legacy code - hard to test
def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    workflow = MLWorkflow(config)  # Direct instantiation
    workflow.run()  # Depends on file system, external services
```

### After: Easy to Test
```python
# ✅ Refactored code - easy to test
def main(config_path):
    config = MLConfig.from_file(config_path)  # Can mock
    workflow = ServiceConfigurator.create_ml_workflow(config)  # Can mock
    workflow.run()  # Can mock external dependencies

# Unit test example
def test_main_success(mocker):
    mock_config = mocker.Mock()
    mock_workflow = mocker.Mock()

    mocker.patch('MLConfig.from_file', return_value=mock_config)
    mocker.patch('ServiceConfigurator.create_ml_workflow', return_value=mock_workflow)

    main('fake_config.yaml')

    MLConfig.from_file.assert_called_once_with('fake_config.yaml')
    ServiceConfigurator.create_ml_workflow.assert_called_once_with(mock_config)
    mock_workflow.run.assert_called_once()
```

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting Path Resolution
```python
# ❌ Wrong - manual path handling
config = yaml.safe_load(open(config_file))
data_path = Path(config_file).parent / config['data_path']

# ✅ Correct - automatic path resolution
config = ConfigClass.from_file(config_file)
data_path = config.data_path  # Already resolved
```

### Pitfall 2: Breaking Existing APIs
```python
# ❌ Breaking change
def main(config_file):  # Was: def main(config_dict)
    config = ConfigClass.from_file(config_file)

# ✅ Backward compatible
def main(config):
    if isinstance(config, str):
        config = ConfigClass.from_file(config)
    # ... rest of function
```

### Pitfall 3: Missing Error Context
```python
# ❌ Poor error message
try:
    service.run()
except Exception as e:
    print(f"Error: {e}")

# ✅ Rich error context
try:
    service.run()
except Exception as e:
    logger.error(f"Workflow execution failed for config {config_file}: {e}")
    logger.debug("Config details:", extra={"config": config.model_dump()})
    raise
```

## Performance Improvements

### Config Loading Optimization
```python
# ConfigClass.from_file() automatically:
# - Caches parsed YAML for repeated loads
# - Resolves relative paths to absolute
# - Validates types at load time
# - Provides rich error messages
```

### Service Reuse
```python
# ServiceConfigurator reuses expensive resources:
# - Logger instances
# - Validation schemas
# - Database connections (when applicable)
# - Cached computations
```

## Integration Patterns

### CLI Integration
```python
# habit/cli.py
def main():
    # ... argument parsing ...

    if args.command == "ml":
        from .cli_commands.commands.cmd_ml import main as ml_main
        ml_main(args.config)
    elif args.command == "habitat":
        from .cli_commands.commands.cmd_habitat import main as habitat_main
        habitat_main(args.config)
```

### Config File Discovery
```python
# Automatic config file discovery
def find_config_file(base_name: str) -> Path:
    """Find config file in standard locations."""
    candidates = [
        Path(f"{base_name}.yaml"),
        Path(f"{base_name}.yml"),
        Path("config") / f"{base_name}.yaml",
        Path("configs") / f"{base_name}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find config file for {base_name}")
```

## Migration Workflow

### Step-by-Step Migration Process

1. **Identify Target Command**
   ```bash
   # Check current migration status
   cat habit/cli_commands/COMMANDS_MIGRATION_STATUS.md
   ```

2. **Analyze Current Implementation**
   ```python
   # Look for yaml.safe_load calls
   grep -r "yaml.safe_load" habit/cli_commands/commands/
   ```

3. **Create/Update Config Class**
   ```python
   # Add to appropriate config_schemas.py
   class NewConfig(BaseConfig):
       field1: str
       field2: int = 42
   ```

4. **Add ServiceConfigurator Method**
   ```python
   # Add to service_configurator.py
   @staticmethod
   def create_new_service(config: NewConfig):
       return NewService(config, logger=cls._create_logger())
   ```

5. **Refactor Command**
   ```python
   # Apply standard pattern
   config = NewConfig.from_file(config_file)
   service = ServiceConfigurator.create_new_service(config)
   service.run()
   ```

6. **Update Tests**
   ```python
   # Add unit tests following new pattern
   def test_new_command():
       # Test config loading, service creation, execution
   ```

7. **Update Documentation**
   ```markdown
   # Update COMMANDS_MIGRATION_STATUS.md
   - ✅ cmd_new.py - Migrated to standard pattern
   ```