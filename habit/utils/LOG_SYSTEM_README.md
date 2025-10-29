# HABIT Project Logging System

## Overview

The HABIT project uses a centralized logging system that ensures consistent log output across all modules. This document explains how to use the logging system correctly.

## Key Design Principles

1. **Single log file per run**: Each application run creates ONE log file, not multiple log files or log folders
2. **Hierarchical logger structure**: All loggers are organized under the `habit` namespace
3. **Consistent formatting**: Console and file outputs have consistent, readable formats
4. **No duplicate logs**: The system prevents duplicate log entries

## Log File Location

Logs are saved directly in the output directory as `processing.log` (or with a custom name):

```
output_dir/
├── processing.log          # Single log file (NOT logs/processing.log)
├── results/
└── other_output_files/
```

**Important**: No `logs/` subfolder is created anymore.

## Usage Guide

### For Main Scripts and CLI Commands

When initializing a new application or script, use `setup_logger()`:

```python
from habit.utils.log_utils import setup_logger

# Setup logger with output directory
logger = setup_logger(
    name='preprocessing',  # Module name
    output_dir=Path('/output'),  # Where to save log file
    log_filename='processing.log',  # Log filename (default: 'processing.log')
    level=logging.INFO  # Log level
)

logger.info('Application started')
```

### For Modules and Utilities

For modules that don't initialize logging themselves, use `get_module_logger()`:

```python
from habit.utils.log_utils import get_module_logger

# Get module logger (will inherit configuration from main logger)
logger = get_module_logger(__name__)

def my_function():
    logger.debug('Processing data...')
    logger.info('Task completed')
```

### For Classes

```python
from habit.utils.log_utils import get_module_logger

class MyClass:
    def __init__(self):
        self.logger = get_module_logger(__name__)
    
    def process(self):
        self.logger.info('Starting process')
```

## Logger Hierarchy

All loggers follow a hierarchical structure:

```
habit (root)
├── habit.preprocessing
│   ├── habit.preprocessing.resample
│   ├── habit.preprocessing.n4_correction
│   └── habit.preprocessing.zscore
├── habit.habitat
│   ├── habit.habitat.clustering
│   └── habit.habitat.features
├── habit.cli
│   ├── habit.cli.preprocess
│   └── habit.cli.habitat
└── habit.scripts
    └── habit.scripts.app_preprocessing
```

## Migration from Old System

### Before (❌ Old way - DO NOT USE)

```python
import logging

# Manual configuration - creates duplicate handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### After (✅ New way - USE THIS)

```python
from habit.utils.log_utils import get_module_logger

# Simple and consistent
logger = get_module_logger(__name__)
```

## Common Patterns

### Pattern 1: Script with Output Directory

```python
from pathlib import Path
from habit.utils.log_utils import setup_logger

def main(config_path):
    # Read config to get output directory
    config = load_config(config_path)
    output_dir = Path(config['out_dir'])
    
    # Setup logging
    logger = setup_logger('myapp', output_dir)
    
    logger.info('Starting processing...')
    # ... rest of code
```

### Pattern 2: Module in Core Library

```python
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

class Preprocessor:
    def __init__(self):
        # Logger is already configured by main script
        logger.info('Initializing preprocessor')
    
    def process(self, data):
        logger.debug(f'Processing {len(data)} items')
```

### Pattern 3: CLI Command

```python
from habit.utils.log_utils import get_module_logger

def run_command(config_path):
    # Use module logger for CLI commands
    logger = get_module_logger('cli.mycommand')
    
    logger.info('Running command...')
    # Main processing class will setup its own logger
    processor = Processor(config_path)
    processor.run()
```

## Log Levels

Use appropriate log levels:

- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages (default)
- `WARNING`: Warning messages for potentially problematic situations
- `ERROR`: Error messages for serious problems
- `CRITICAL`: Critical errors that may cause program termination

## Disable External Library Logging

To reduce noise from external libraries:

```python
from habit.utils.log_utils import disable_external_loggers

disable_external_loggers()  # Silences verbose libraries
```

## Troubleshooting

### Problem: Seeing duplicate log messages

**Solution**: Make sure you're using `get_module_logger()` instead of manually configuring logging.

### Problem: Log file not created

**Solution**: Ensure `setup_logger()` is called with an `output_dir` parameter.

### Problem: Logs from different modules not appearing

**Solution**: Check that the module uses `get_module_logger(__name__)` and the main script has called `setup_logger()`.

## Best Practices

1. ✅ Always use `setup_logger()` in main scripts/CLI commands
2. ✅ Always use `get_module_logger(__name__)` in modules
3. ✅ Never call `logging.basicConfig()` directly
4. ✅ Never create multiple FileHandlers manually
5. ✅ Use meaningful logger names (e.g., 'preprocessing', 'habitat')
6. ✅ Log at appropriate levels (INFO for general, DEBUG for details)

## Examples from HABIT Codebase

### BatchProcessor (preprocessing)

```python
class BatchProcessor:
    def __init__(self, config_path):
        # Setup logging at initialization
        self.logger = setup_logger(
            name='preprocessing',
            output_dir=self.output_root,
            log_filename='processing.log'
        )
```

### HabitatAnalysis (habitat analysis)

```python
class HabitatAnalysis:
    def _setup_logging(self, log_level):
        self.logger = setup_logger(
            name='habitat',
            output_dir=self.out_dir,
            log_filename='habitat_analysis.log',
            level=getattr(logging, log_level.upper())
        )
```

### N4BiasFieldCorrection (module)

```python
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

class N4BiasFieldCorrection(BasePreprocessor):
    def __call__(self, data):
        logger.info('Applying N4 bias field correction')
        # ... processing
```

## Summary

The new logging system provides:
- ✅ Single, clean log file per run
- ✅ No duplicate log folders or files
- ✅ Consistent formatting across all modules
- ✅ Easy to use and maintain
- ✅ Thread-safe operation
- ✅ Proper hierarchy for debugging

Follow the patterns in this guide, and logging will "just work" throughout your HABIT project.

