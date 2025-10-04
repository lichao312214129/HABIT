# Documentation Updates Summary

This document summarizes the modifications and improvements made to the HABIT project documentation.

## Modification Date
2025-10-04

## Modified Documents

### 1. `doc/app_dcm2nii.md`
**Changes**:
- Added a note: This function is not a standalone application script but is used as part of the preprocessing pipeline.
- Updated example: Changed the standalone script example to an example of usage via `app_image_preprocessing.py`.
- Explicitly stated that `app_image_preprocessing.py` should be referenced for using this functionality.

**Reason for Change**:
- There is no separate dcm2niix application script in the codebase.
- The functionality has been integrated into the preprocessing pipeline.

### 2. `doc/app_getting_habitat_map.md`
**Changes**:
- Updated the list of supported clustering algorithms.
  - Added: `mean_shift`, `dbscan`, `affinity_propagation`.
  - Clarified that habitat clustering mainly supports `kmeans` and `gmm`.
- Updated cluster number selection methods.
  - Added: `aic`, `bic` (for GMM).
  - Clarified the applicable algorithms for each method.
- Added description for the `mode` parameter (training/testing).
- Added description for the `min_clusters` parameter.
- Improved the configuration example.

**Reason for Change**:
- The actual code supports more clustering algorithms.
- New parameters in the configuration file were not documented.
- The applicable scenarios for different parameters needed clarification.

### 3. `doc/app_image_preprocessing.md`
**Changes**:
- Corrected a typo in the usage section (`ython` → `python`).
- Added a shorthand command example (`-c` parameter).
- Simplified the registration parameter description, removing `metric` and `optimizer` parameters that are not actually used.
- Added a note that ANTsPy automatically selects optimal parameters.
- Updated the list of supported transformation types.
  - Added: `SyNBold`, `SyNBoldAff`, `SyNAggro`, `TVMSQ`.
- Updated the complete configuration example with a more practical setup.

**Reason for Change**:
- Some parameters listed in the document were not used in the actual code.
- Newly added registration methods needed to be included.
- The configuration example needed to be more aligned with practical use cases.

### 4. `doc/app_extracting_habitat_features.md`
**Changes**:
- Improved the description of feature types.
  - Clarified the specific content of `non_radiomics` features.
  - Detailed the difference between `whole_habitat` and `each_habitat`.
  - Added a detailed description of `msi` (multiregional spatial interaction matrix features).
  - Added the `ith_score` (Intratumoral Heterogeneity Index) feature type.
- Updated the complete configuration example to include all supported feature types.
- Enhanced the notes section with more practical advice.

**Reason for Change**:
- The original document's description of feature types was not detailed enough.
- The `ith_score` feature type was missing.
- The calculation methods and application scenarios for each feature type needed to be clarified.

### 5. `INSTALL.md`
**Changes**:
- Updated Python version requirements (3.8 or higher, 3.8-3.10 recommended).
- Added a note about optional R language dependency.
- Completed the list of core dependencies.
  - Added: `lightgbm`, `scipy`, `statsmodels`, `mrmr_selection`, `pingouin`, `shap`, `lifelines`, `opencv-python`, `trimesh`, `torch`.
  - Updated table format for clarity.
- Updated optional dependency notes.
  - Added AutoGluon instructions.
  - Added R language interface (rpy2) instructions.
  - Updated torch installation instructions.
- Improved the installation verification section.
  - Updated verification commands to use existing modules.
  - Added more feature verification examples.
- Modified configuration file setup instructions.
  - Recommended directly modifying configuration files in the `config` folder.
- Added solutions for R-related errors.
- Updated commands for environment cleanup and recreation.
- Added a "Next Steps" guide.

**Reason for Change**:
- The dependency list was incomplete.
- The Python version information was inaccurate.
- R-related instructions were needed (for some feature selection methods).
- Installation verification commands needed updating.
- The troubleshooting section needed to be more comprehensive.

## Documents Not Modified but Recommended for Review

### `doc/app_habitat_test_retest.md`
- The functionality described in this document corresponds to the `app_habitat_test_retest_mapper.py` script.
- It is recommended to verify that the parameters in the document are consistent with the actual code.

### `doc/app_icc_analysis.md`
- The configuration file format needs to be verified for consistency with the actual code.
- It is recommended to check if the ICC analysis types and parameter descriptions are accurate.

### `doc/app_of_machine_learning.md`
- It is recommended to verify the completeness of the feature selection methods and machine learning model lists.
- Check if the parameter descriptions are consistent with the actual code.

### `doc/app_model_comparison_plots.md`
- It is recommended to verify the configuration file format and parameter descriptions.

## Modification Principles

1.  **Accuracy**: All parameter and function descriptions are based on the actual code.
2.  **Completeness**: Added missing parameter and function descriptions.
3.  **Practicality**: Provided configuration examples that are closer to real-world use.
4.  **Consistency**: Ensured that the documentation is consistent with the configuration files and code.
5.  **Clarity**: Used a clearer structure and explanations.

## Recommendations

1.  **Regular Synchronization**: It is recommended to update the documentation promptly after code updates.
2.  **Version Management**: Consider adding version numbers and update dates to the documents.
3.  **Example Validation**: Periodically test the example code and configurations in the documentation.
4.  **User Feedback**: Collect user feedback to continuously improve the quality of the documentation.

## To-Do Items

1.  Add more practical use cases and tutorials.
2.  Create a Frequently Asked Questions (FAQ) document.
3.  Add detailed troubleshooting steps and screenshots.
4.  Supplement with performance optimization recommendations.
5.  Add API reference documentation (if needed).

---

If you have any questions or suggestions, please contact the documentation maintenance team.
