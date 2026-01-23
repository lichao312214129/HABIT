---
name: habit-refactoring
description: Refactor habit package commands to use standard patterns with ConfigClass.from_file() and ServiceConfigurator. Use when working with habit CLI commands, migrating legacy code, or implementing new commands following the established refactoring patterns.
---

# Habit Package Refactoring

## Overview

This skill guides the refactoring of the habit package to follow standardized patterns using `ConfigClass.from_file()` for configuration loading and `ServiceConfigurator` for dependency injection.

## Standard Refactoring Pattern

### Before (Legacy Pattern)
```python
# ❌ Legacy pattern - avoid this
import yaml

def main(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    # Direct instantiation
    service = SomeService(config)
    service.run()
```

### After (Standard Pattern)
```python
# ✅ Standard pattern - use this
def main(config_file):
    config = ConfigClass.from_file(config_file)
    service = ServiceConfigurator.create_service(config)
    service.run()
```

## Refactoring Steps

### 1. Update Configuration Loading
Replace direct YAML loading with typed configuration classes:

```python
# Old
with open(config_file) as f:
    config = yaml.safe_load(f)

# New
config = ConfigClass.from_file(config_file)  # Type-safe, path resolution included
```

### 2. Use Service Configurator
Replace direct service instantiation with dependency injection:

```python
# Old
service = SomeService(config, logger=logger, ...)

# New
service = ServiceConfigurator.create_service(config)  # All dependencies injected
```

### 3. Update Error Handling
Use standardized error handling patterns:

```python
# Old
try:
    # code
except Exception as e:
    print(f"Error: {e}")

# New
try:
    # code
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise
```

## Module-Specific Patterns

### Machine Learning Commands (cmd_ml.py)
```python
# Use MLConfig and create_ml_workflow
config = MLConfig.from_file(config_path)
workflow = ServiceConfigurator.create_ml_workflow(config)
workflow.run()
```

### Habitat Analysis Commands (cmd_habitat.py)
```python
# Use HabitatAnalysisConfig and create_habitat_analysis
config = HabitatAnalysisConfig.from_file(config_file)
service = ServiceConfigurator.create_habitat_analysis(config)
service.run()
```

### Preprocessing Commands (cmd_preprocess.py)
```python
# Use PreprocessingConfig and create_batch_processor
config = PreprocessingConfig.from_file(config_path)
processor = ServiceConfigurator.create_batch_processor(config)
processor.run()
```

## Common Refactoring Tasks

### Adding New Commands
1. Create command file in `habit/cli_commands/commands/`
2. Implement main function following standard pattern
3. Add to CLI routing in `habit/cli.py`
4. Update `COMMANDS_MIGRATION_STATUS.md`

### Migrating Legacy Commands
1. Identify legacy YAML loading: `yaml.safe_load()`
2. Find corresponding ConfigClass (e.g., `MLConfig`, `HabitatAnalysisConfig`)
3. Replace direct service instantiation with ServiceConfigurator method
4. Update error handling to use proper logging

## Quality Checks

### ✅ Migration Status Indicators
- [ ] Uses `ConfigClass.from_file(config_file)` - type-safe config loading
- [ ] Uses `ServiceConfigurator.create_*()` - dependency injection
- [ ] Proper error handling with logging
- [ ] No direct YAML loading
- [ ] No direct service instantiation with config dicts

### ❌ Anti-patterns to Avoid
- Direct `yaml.safe_load()` calls
- Manual path resolution
- Direct instantiation: `Service(config_dict)`
- Print statements for errors
- Hardcoded file paths

## Examples

### Example 1: Basic Command Migration
```python
# habit/cli_commands/commands/cmd_example.py

def main(config_file: str):
    """Example command following standard pattern."""
    # ✅ Standard config loading
    config = ExampleConfig.from_file(config_file)

    # ✅ Standard service creation
    service = ServiceConfigurator.create_example_service(config)

    # ✅ Standard execution
    service.run()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
```

### Example 2: Complex Workflow Migration
```python
# habit/cli_commands/commands/cmd_complex.py

def main(config_file: str, workflow_type: str = "default"):
    """Complex workflow command."""
    config = ComplexConfig.from_file(config_file)

    if workflow_type == "kfold":
        workflow = ServiceConfigurator.create_kfold_workflow(config)
    else:
        workflow = ServiceConfigurator.create_complex_workflow(config)

    workflow.run()
```

## Additional Resources

- For migration status, see [COMMANDS_MIGRATION_STATUS.md](../../cli_commands/COMMANDS_MIGRATION_STATUS.md)
- For configuration patterns, see `habit/core/common/config_base.py`
- For service patterns, see `habit/core/common/service_configurator.py`