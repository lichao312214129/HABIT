# HABIT Package Import Robustness Guide

## Overview

The HABIT package now includes a robust import fault tolerance mechanism, ensuring that the package remains usable even if some modules fail to import. This mechanism provides:

1.  **Graceful Error Handling** - Import failures do not crash the entire package.
2.  **State Tracking** - You can query which modules are available and which have failed.
3.  **Warning System** - Import error messages are displayed automatically.
4.  **Utility Tools** - Provides functions for checking and diagnosing.

## Basic Usage

### Importing the Package

```python
import habit

# Check package version
print(f"HABIT version: {habit.__version__}")

# Check available modules
available_modules = habit.get_available_modules()
print(f"Available modules: {list(available_modules.keys())}")

# Check for import errors
import_errors = habit.get_import_errors()
if import_errors:
    print(f"Import errors: {import_errors}")
```

### Checking Module Availability

```python
import habit

# Check if a specific module is available
if habit.is_module_available('HabitatAnalysis'):
    analyzer = habit.HabitatAnalysis()
    # Use the analyzer...
else:
    print("HabitatAnalysis is not available")
    print(f"Error: {habit.get_import_errors().get('HabitatAnalysis')}")

if habit.is_module_available('Modeling'):
    model = habit.Modeling()
    # Use the model...
else:
    print("Modeling is not available")
```

## Advanced Features

### Using ImportManager

The `ImportManager` class provides more advanced import management features:

```python
from habit.utils.import_utils import ImportManager

# Create an import manager
manager = ImportManager()

# Safely import a module
numpy = manager.safe_import('numpy', alias='np')
pandas = manager.safe_import('pandas', alias='pd')

# Safely import a class
rf_classifier = manager.safe_import('sklearn.ensemble', 'RandomForestClassifier', 'RFC')

# Batch import
imports = [
    ('matplotlib.pyplot', None, 'plt'),
    ('seaborn', None, 'sns'),
    ('sklearn.metrics', 'accuracy_score', 'acc_score'),
]

results = manager.safe_import_multiple(imports)

# Check import status
manager.print_import_status(verbose=True)

# Get error information
errors = manager.get_import_errors()
warnings = manager.get_import_warnings()
```

### Dependency Checking

```python
from habit.utils.import_utils import check_dependencies

# Check required and optional dependencies
required_modules = ['numpy', 'pandas', 'sklearn']
optional_modules = ['matplotlib', 'seaborn', 'plotly']

status = check_dependencies(required_modules, optional_modules)

for module, available in status.items():
    if available:
        print(f"✓ {module} is available")
    else:
        print(f"✗ {module} is not available")
```

### Querying Module Information

```python
from habit.utils.import_utils import get_module_info

# Get detailed module information
modules_to_check = ['numpy', 'pandas', 'nonexistent_module']

for module in modules_to_check:
    info = get_module_info(module)
    print(f"\n{module}:")
    print(f"  Available: {info['available']}")
    print(f"  Version: {info['version']}")
    print(f"  Path: {info['path']}")
    if info['error']:
        print(f"  Error: {info['error']}")
```

### Using Decorators

```python
from habit.utils.import_utils import safe_import_decorator

# Safely import using a decorator
@safe_import_decorator('matplotlib.pyplot', alias='plt', default_value=None)
def plot_data(data, plt):
    if plt is None:
        print("matplotlib.pyplot is not available, skipping plot")
        return
    
    plt.plot(data)
    plt.show()

# Call the function
plot_data([1, 2, 3, 4, 5])
```

## Error Handling Best Practices

### 1. Check Module Availability

```python
import habit

def safe_analysis():
    if not habit.is_module_available('HabitatAnalysis'):
        print("HabitatAnalysis is not available")
        return None
    
    try:
        analyzer = habit.HabitatAnalysis()
        return analyzer
    except Exception as e:
        print(f"Error creating analyzer: {e}")
        return None
```

### 2. Provide Alternatives

```python
from habit.utils.import_utils import ImportManager

def get_plotting_backend():
    manager = ImportManager()
    
    # Try different plotting backends
    backends = [
        ('matplotlib.pyplot', None, 'plt'),
        ('plotly.express', None, 'px'),
        ('seaborn', None, 'sns'),
    ]
    
    for module_path, class_name, alias in backends:
        backend = manager.safe_import(module_path, class_name, alias)
        if backend is not None:
            print(f"Using {alias} for plotting")
            return backend
    
    print("No plotting backend available")
    return None
```

### 3. Conditional Feature Enablement

```python
import habit

class FeatureProcessor:
    def __init__(self):
        self.has_ml = habit.is_module_available('Modeling')
        self.has_analysis = habit.is_module_available('HabitatAnalysis')
    
    def process_features(self, data):
        if self.has_analysis:
            # Process using HabitatAnalysis
            analyzer = habit.HabitatAnalysis()
            return analyzer.process(data)
        else:
            # Use basic processing
            return self.basic_process(data)
    
    def train_model(self, X, y):
        if self.has_ml:
            # Train using Modeling
            model = habit.Modeling()
            return model.train(X, y)
        else:
            print("Machine learning module not available")
            return None
```

## Testing Import Robustness

Run the test script to verify the import fault tolerance mechanism:

```bash
python scripts/test_import_robustness.py
```

This script tests:
- Basic import functionality
- Module availability checks
- Import utility functions
- Graceful failure handling

## FAQ

### Q: How do I know which modules failed to import?

A: Use `habit.get_import_errors()` to get detailed error information:

```python
import habit

errors = habit.get_import_errors()
for module, error in errors.items():
    print(f"{module}: {error}")
```

### Q: How can I force a re-import of a module?

A: Use Python's `importlib.reload()`:

```python
import importlib
import habit

# Reload the module
importlib.reload(habit)
```

### Q: How can I disable import warnings?

A: Use Python's `warnings` module:

```python
import warnings
warnings.filterwarnings("ignore", category=ImportWarning)

import habit  # No import warnings will be displayed
```

### Q: How can I check for a specific version of a module?

A: Use the `get_module_info()` function:

```python
from habit.utils.import_utils import get_module_info

info = get_module_info('numpy')
if info['available'] and info['version']:
    print(f"NumPy version: {info['version']}")
```

## Summary

The import fault tolerance mechanism of the HABIT package provides:

1.  **Reliability** - The package works even if some modules are missing.
2.  **Transparency** - It's clear which features are available and which are not.
3.  **Flexibility** - Functionality can be adjusted based on available modules.
4.  **Debuggability** - Detailed error messages and state tracking.

This mechanism ensures the stability and usability of the HABIT package in various environments.
