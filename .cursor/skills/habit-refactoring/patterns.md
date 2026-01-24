# Habit Refactoring Patterns - Detailed Reference

## Configuration Classes Available

### Core Configuration Classes
- `MLConfig` - Machine learning workflows
- `HabitatAnalysisConfig` - Habitat analysis pipelines
- `PreprocessingConfig` - Image/data preprocessing
- `ModelComparisonConfig` - Model comparison workflows
- `FeatureExtractionConfig` - Feature extraction pipelines

### Usage Pattern
```python
from habit.core.machine_learning.config_schemas import MLConfig

config = MLConfig.from_file("path/to/config.yaml")
# config is now typed, validated, and paths are resolved
```

## ServiceConfigurator Methods

### Available Factory Methods
- `create_ml_workflow(config: MLConfig)` → ML workflow service
- `create_habitat_analysis(config: HabitatAnalysisConfig)` → Habitat analysis service
- `create_batch_processor(config: PreprocessingConfig)` → Preprocessing service
- `create_model_comparison(config: ModelComparisonConfig)` → Comparison service
- `create_feature_extractor(config: FeatureExtractionConfig)` → Feature extraction service
- `create_kfold_workflow(config: MLConfig)` → K-fold validation workflow

### Usage Pattern
```python
from habit.core.common.service_configurator import ServiceConfigurator

service = ServiceConfigurator.create_ml_workflow(config)
# All dependencies (logger, validators, etc.) are injected automatically
```

## Command Structure Template

### Standard Command Template
```python
#!/usr/bin/env python3
"""
Command for [describe what it does].

Usage:
    python -m habit.cli_commands.commands.cmd_example config.yaml
"""

import sys
from pathlib import Path

# Import appropriate config class
from habit.core.[module].config_schemas import ConfigClass

# Import service configurator
from habit.core.common.service_configurator import ServiceConfigurator


def main(config_file: str) -> None:
    """
    Main entry point for the command.

    Args:
        config_file: Path to configuration YAML file
    """
    try:
        # ✅ Standard config loading
        config = ConfigClass.from_file(config_file)

        # ✅ Standard service creation
        service = ServiceConfigurator.create_service(config)

        # ✅ Standard execution
        service.run()

    except Exception as e:
        # Use proper logging (will be injected by ServiceConfigurator)
        logger = service.logger if 'service' in locals() else None
        if logger:
            logger.error(f"Command execution failed: {e}")
        else:
            print(f"Error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cmd_example.py <config_file>", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
```

## Migration Examples

### Before: Legacy cmd_ml.py Pattern
```python
# ❌ OLD - Legacy pattern
import yaml
from habit.core.machine_learning.workflows import MLWorkflow

def main(config_path):
    # Direct YAML loading
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Direct instantiation
    workflow = MLWorkflow(config_dict)
    workflow.run()
```

### After: Standard cmd_ml.py Pattern
```python
# ✅ NEW - Standard pattern
from habit.core.machine_learning.config_schemas import MLConfig
from habit.core.common.service_configurator import ServiceConfigurator

def main(config_path):
    # Type-safe config loading
    config = MLConfig.from_file(config_path)

    # Dependency injection
    workflow = ServiceConfigurator.create_ml_workflow(config)
    workflow.run()
```

## Error Handling Patterns

### Standard Error Handling
```python
def main(config_file: str) -> None:
    try:
        config = ConfigClass.from_file(config_file)
        service = ServiceConfigurator.create_service(config)
        service.run()

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {config_file}")
        raise

    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}")
        raise
```

## Testing Patterns

### Unit Test Template
```python
import pytest
from pathlib import Path
from habit.core.[module].config_schemas import ConfigClass
from habit.core.common.service_configurator import ServiceConfigurator


class TestCommand:
    """Test cases for command functionality."""

    def test_config_loading(self, tmp_path):
        """Test that configuration loads correctly."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
        # Test configuration
        param1: value1
        param2: value2
        """)

        config = ConfigClass.from_file(str(config_file))
        assert config.param1 == "value1"

    def test_service_creation(self, config):
        """Test that service is created correctly."""
        service = ServiceConfigurator.create_service(config)
        assert service is not None
        assert hasattr(service, 'run')

    def test_full_workflow(self, config):
        """Test complete command execution."""
        service = ServiceConfigurator.create_service(config)
        # Mock any external dependencies
        service.run()  # Should not raise
```

## Common Migration Issues

### Issue 1: Missing Config Class
**Problem**: No ConfigClass exists for the module
**Solution**: Create new config class following existing patterns in `config_schemas.py`

### Issue 2: Missing ServiceConfigurator Method
**Problem**: No factory method for the service
**Solution**: Add method to `ServiceConfigurator` following existing patterns

### Issue 3: Complex Legacy Logic
**Problem**: Legacy code has complex initialization logic
**Solution**: Extract logic into service class, use ServiceConfigurator for instantiation

### Issue 4: Circular Dependencies
**Problem**: Config classes depend on services, services depend on configs
**Solution**: Use forward references and lazy loading in ServiceConfigurator

## Performance Considerations

### Config Loading
- ConfigClass.from_file() includes path resolution and validation
- Cached loading for repeated calls
- Type validation prevents runtime errors

### Service Creation
- ServiceConfigurator handles dependency injection
- Reuses common services (logger, validators)
- Lazy initialization for expensive resources

## Integration with CLI

### Adding New Commands
1. Create `cmd_[name].py` in `habit/cli_commands/commands/`
2. Follow standard template above
3. Add routing in `habit/cli.py`:

```python
# In habit/cli.py
elif args.command == "new_command":
    from .cli_commands.commands.cmd_new import main
    main(args.config)
```

4. Update `COMMANDS_MIGRATION_STATUS.md`
5. Add tests in `tests/`

## Validation Checklist

### Pre-Migration
- [ ] Identify config schema requirements
- [ ] Check for existing ConfigClass
- [ ] Verify ServiceConfigurator method exists
- [ ] Review legacy error handling

### During Migration
- [ ] Replace yaml.safe_load() with ConfigClass.from_file()
- [ ] Replace direct instantiation with ServiceConfigurator
- [ ] Update error handling to use logging
- [ ] Maintain backward compatibility if needed

### Post-Migration
- [ ] Run existing tests
- [ ] Test configuration validation
- [ ] Verify service dependencies are injected
- [ ] Check logging output
- [ ] Update documentation