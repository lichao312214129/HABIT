# Changelog

All notable changes to the HABIT public Python API are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Stable top-level namespace: ``import habit`` exposes pipeline runners, config
  classes, and utilities via lazy loading (see ``habit.api.registry``).
- Public runners: ``run_preprocess``, ``run_dicom_sort``, ``run_habitat_analysis``,
  ``run_feature_extraction``, ``run_radiomics``, ``run_ml``, ``run_kfold``,
  ``run_model_comparison``, ``run_icc_analysis``.
- Public config classes: ``PreprocessingConfig``, ``DicomSortConfig``,
  ``HabitatAnalysisConfig``, ``FeatureExtractionConfig``, ``RadiomicsConfig``,
  ``MLConfig``, ``ModelComparisonConfig``, ``TestRetestConfig``, ``ICCConfig``.
- Helpers: ``apply_habitat_cli_overrides``, ``apply_ml_mode_override``,
  ``setup_logger``, ``is_available``.
- ``habit.__version__`` sourced from ``habit._version``.
- API contract tests under ``tests/api/`` (golden MSI/ITH, pipeline smoke,
  CLI–API parity) and CI workflow ``.github/workflows/tests.yml``.

### Notes

- Internal modules under ``habit.core.*`` remain implementation details. Prefer
  the top-level ``habit`` imports for new integrations.
- Deep paths such as ``habit.core.preprocessing.run.run_preprocess_from_config``
  continue to work unchanged.
