# Image preprocessing module

Batch pipelines are driven by YAML (`PreprocessingConfig`). Step names under `Preprocessing` must match `PreprocessorFactory` registration names (`sort_dicom`, `dcm2nii`, `n4_correction`, `resample`, `registration`, `zscore_normalization`, `histogram_standardization`, `adaptive_histogram_equalization`, plus your custom registrations).

**Authoritative user documentation:** `docs/source/user_guide/image_preprocessing_zh.rst` and `docs/source/configuration_zh.rst` (preprocessing section).

## YAML sketch (abbreviated)

```yaml
data_dir: ./files_preprocessing.yaml
out_dir: ./preprocessed
auto_select_first_file: true

Preprocessing:
  sort_dicom:
    images: [dicom]
    dcm2niix_path: ./dcm2niix.exe
    filename_format: "%n_%g_%x/%s_%d/%r_%o.dcm"
    output_dir: ./sorted_dicom

  n4_correction:
    images: [t1, t2]
    num_fitting_levels: 4
    num_iterations: [50, 50, 50, 50]
    convergence_threshold: 0.001
    shrink_factor: 4

  resample:
    images: [t1, t2]
    target_spacing: [1.0, 1.0, 1.0]
    img_mode: bilinear

  zscore_normalization:
    images: [t1, t2]
    only_inmask: false
    clip_values: [-3, 3]

  histogram_standardization:
    images: [t1, t2]
    percentiles: [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    target_min: 0.0
    target_max: 100.0
```

## Direct `PreprocessorFactory` usage

Factory kwargs mirror each preprocessor `__init__` (plus `keys` for modalities):

```python
from habit.core.preprocessing import PreprocessorFactory

resample = PreprocessorFactory.create(
    "resample",
    keys=["t1", "t2"],
    target_spacing=[1.0, 1.0, 1.0],
    img_mode="bilinear",
)

zscore = PreprocessorFactory.create(
    "zscore_normalization",
    keys=["t1", "t2"],
    only_inmask=True,
    mask_key="mask_t1",  # must exist in the `data` dict when only_inmask is True
    clip_values=(-3.0, 3.0),
)
```

## Registration stage notes

- Floating series are **all entries in `images` except `fixed_image`**. Do **not** add a `moving_images` field to YAML; it is not read by `RegistrationPreprocessor` and may pollute ANTs kwargs when `backend: ants`.
- **`backend`**: `ants` (default, ANTsPy) or `simpleitk` (SimpleITK `ImageRegistrationMethod`; no ANTsPy required).
- When `backend: ants`, registration requires **ANTsPy**. When `backend: simpleitk`, only **SimpleITK** is used (same stack as resample / N4).
- **SimpleITK-only YAML tuning keys** (stripped when `backend: ants`): `number_of_histogram_bins`, `metric_sampling_percentage`, `shrink_factors_per_level`, `smoothing_sigmas_per_level`, `learning_rate`, `number_of_iterations`, `bspline_mesh_size`, `bspline_order`. See `habit/core/preprocessing/registration.py` (`_SITK_OPTION_KEYS`) and `docs/source/user_guide/image_preprocessing_zh.rst`.

## Custom preprocessors

Subclass `BasePreprocessor`, register with `@PreprocessorFactory.register("step_name")`, and import the module from `habit.core.preprocessing` (or `__init__.py`) so the decorator runs. See `custom_preprocessor_template.py`.
