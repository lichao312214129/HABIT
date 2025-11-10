# HABIT Test Suite

This directory contains test files for the HABIT (Habitat Analysis: Biomedical Imaging Toolkit) package.

## Test Structure

### Debug Scripts (Quick Testing)
These scripts simulate CLI commands for quick debugging:

- `debug_preprocess.py` - Test image preprocessing pipeline
- `debug_habitat.py` - Test habitat analysis
- `debug_extract_features.py` - Test feature extraction
- `debug_radiomics.py` - Test radiomics extraction
- `debug_ml.py` - Test machine learning pipeline
- `debug_kfold.py` - Test k-fold cross validation
- `debug_icc.py` - Test ICC analysis
- `debug_test_retest.py` - Test test-retest reliability
- `debug_compare.py` - Test model comparison

### Unit Tests (Comprehensive Testing)
These files contain pytest-based unit tests:

- `test_preprocessing.py` - Unit tests for preprocessing module
- `test_habitat_analysis.py` - Unit tests for habitat analysis
- `test_machine_learning.py` - Unit tests for ML module
- `test_utils.py` - Unit tests for utility functions
- `test_cli.py` - Unit tests for CLI commands

## Running Tests

### Run a specific debug script:
```bash
cd tests
python debug_preprocess.py
```

### Run all unit tests:
```bash
# From project root
pytest tests/ -v

# Or from tests directory
cd tests
pytest -v
```

### Run specific test file:
```bash
pytest tests/test_preprocessing.py -v
```

### Run specific test class:
```bash
pytest tests/test_preprocessing.py::TestN4Correction -v
```

### Run specific test function:
```bash
pytest tests/test_preprocessing.py::TestN4Correction::test_n4_correction_basic -v
```

### Run with coverage:
```bash
pytest tests/ --cov=habit --cov-report=html
```

## Configuration Files

Before running debug scripts, ensure you have appropriate configuration files in the `demo_image_data` directory:

- `config_image_preprocessing.yaml` - Preprocessing configuration
- `config_habitat_analysis.yaml` - Habitat analysis configuration
- `config_feature_extraction.yaml` - Feature extraction configuration
- `config_radiomics.yaml` - Radiomics extraction configuration
- `config_ml.yaml` - Machine learning configuration
- `config_kfold.yaml` - K-fold CV configuration
- `config_icc.yaml` - ICC analysis configuration
- `config_test_retest.yaml` - Test-retest configuration
- `config_compare.yaml` - Model comparison configuration

## Notes

1. Debug scripts use absolute paths - update them according to your local setup
2. Unit tests are placeholders - implement based on actual module functionality
3. Add more test cases as you develop new features
4. Ensure all tests pass before committing code changes
5. Maintain test coverage above 80% for critical modules

