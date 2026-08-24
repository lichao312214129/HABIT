Traditional Radiomics
=====================

**Config**: ``feature_types`` includes ``traditional``

**Output**: ``raw_image_radiomics.csv``

Definition
----------

PyRadiomics features extracted from each **preprocessed image modality** within a
tumor ROI. The ROI is a **binary mask** derived from the habitat label map
(voxels with label :math:`\ge 1` → foreground); anatomical ``masks/`` files are
not used. Habitat labels are otherwise ignored—features reflect intratumoral
signal and texture on the raw (multi-delay) images.

Implementation
--------------

- Parameter file: ``params_file_of_non_habitat`` (optional; bundled ``roi`` preset
  → ``habit/resources/radiomics/parameter.yaml`` when omitted)
- Code: ``habit/compat/engines/habitat_extraction/habitat_features/builtin_plugins.py``
  (``TraditionalRadiomicsPlugin``) → ``habitat_radiomics.py``

Output columns
--------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Column pattern
     - Description
   * - ``{pyradiomics_feature}_of_{modality}``
     - One column per PyRadiomics feature × image modality under the subject folder
   * - (excluded)
     - Columns whose names contain ``diagnostic`` are dropped before export

Feature definitions
-------------------

HABIT extracts first-order, GLCM, GLRLM, GLSZM, GLDM (IBSI NGLDM), NGTDM,
and 3-D shape features using the PyRadiomics 3.1 formulas, which follow
the Image Biomarker Standardisation Initiative (IBSI;
`Zwanenburg et al., Radiology 2020 <https://doi.org/10.1148/radiol.2020191145>`_).
Texture aggregation is IBSI **3-D averaged** (mean over the 13 unique
3-D angles), not 3-D merged. The same definitions are used by
``habit radiomics``, ``traditional`` / ``each_habitat`` /
``whole_habitat`` feature types, ``supervoxel_radiomics``,
``voxel_radiomics``, and the native C path.

PyRadiomics alignment
~~~~~~~~~~~~~~~~~~~~~

**ROI-level radiomics, voxel-level radiomics, and 3-D shape all match
PyRadiomics 3.1** ``FeatureExtractor.execute()`` on the same image, mask,
label, and settings (``binWidth``, ``voxelArrayShift``, distances, …).

**ROI-level** (``habit radiomics`` / ``traditional``, ``each_habitat``,
``whole_habitat``, ``supervoxel_radiomics`` with default
``union_bin=false``): first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM.
``traditional`` / ``whole_habitat`` call ``execute()`` directly.
``each_habitat`` and default ``supervoxel_radiomics`` use the native C
path with per-label bins; vs ``execute()`` they agree to about
``1e-10`` relative (Energy / TotalEnergy at floating-point ULP).

**Shape** (ROI only): ``original_shape_*`` from PyRadiomics
``execute()`` / ``computeShape``. Shape is a whole-ROI mesh quantity;
there is no per-voxel shape map.

**Voxel-level** (``voxel_radiomics``): first-order and texture via
``execute(..., voxelBased=True)``. The CPU path **is** that extractor
(pre-crop does not change values). Torch / CUDA uses the same formulas;
texture vs CPU sits at about :math:`10^{-15}`, first-order percentiles
at about :math:`10^{-8}` (quantile algorithm). Set
``torch_dtype: float64`` for the closest CPU match.

Gates: ``tests/kernels/test_supervoxel_native_parity.py``,
``tests/kernels/test_ibsi_digital_phantom.py``,
``tests/kernels/test_*_gpu_parity.py``.

These are **not** PyRadiomics and must not be compared to ``execute()``:
``graph``, ``msi``, ``ith_score``, ``volume``, ``local_entropy``, and
supervoxel ``mean`` / ``std`` / ``percentile``.

``supervoxel_radiomics`` with ``union_bin=true`` is a different
discretization (one shared gray scale on the union mask).
``JointAverage`` / ``Autocorrelation`` / ``HighGrayLevel*`` will not
match per-ROI ``execute()``; that is intended, not a formula bug.

Two conventions must be read with the IBSI manual, not against it:

- **Kurtosis** is Pearson kurtosis. IBSI reports *excess* kurtosis
  (Fisher, normal distribution = 0). HABIT / PyRadiomics = IBSI + 3.
- HABIT does **not** extract IBSI families it does not implement:
  GLDZM, local-intensity peak, intensity-volume histogram, Moran's I,
  or Geary's C. Deprecated PyRadiomics shape names (Compactness 1/2,
  Spherical disproportion) are not computed.

IBSI-1 Phase 1 digital phantom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table uses the official IBSI digital phantom (5×4×4 voxels, 2 mm
isotropic, 74-voxel ROI; intensities include 1, 3, 4, 6, 9)
from `theibsi/data_sets <https://github.com/theibsi/data_sets/tree/master/ibsi_1_digital_phantom>`_
(CC-BY-4.0) and the published **dig. phantom / 3-D averaged** (or **3-D**
for GLSZM / NGTDM / NGLDM) reference values in the IBSI reference manual
(`Image features <https://ibsi.readthedocs.io/en/latest/03_Image_features.html>`_).
Settings match Phase 1: no interpolation, ``binWidth=1``,
``symmetricalGLCM=True``, ``distances=[1]``, NGLDM coarseness
:math:`\alpha=0`, ``voxelArrayShift=0``.

``HABIT`` is the native C + CPU-formula value when that family is in the
fast path; shape is the PyRadiomics ``execute`` value used by
``traditional_radiomics``. A regression test loads the same NIfTI pair:
``tests/kernels/test_ibsi_digital_phantom.py``.

.. csv-table:: IBSI-1 Phase 1 digital phantom (3-D averaged texture)
   :file: ibsi_phase1_digital_phantom.csv
   :header-rows: 1
   :widths: 12 28 10 12 12 12 24

See `PyRadiomics Feature Reference <https://pyradiomics.readthedocs.io/en/latest/features.html>`_
for per-feature formulas.
