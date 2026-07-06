# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.
#
"""
Bundled PyRadiomics parameter presets.

These YAML files are the package-internal defaults used when a user omits
``params_file`` for a radiomics-based feature extractor or workflow. Resolve
them through :func:`habit.utils.radiomics_preset_utils.resolve_params_file`.

Preset keys (see :data:`habit.utils.radiomics_preset_utils.PRESET_FILES`):

* ``voxel``          -> ``params_voxel_radiomics.yaml`` (CT R3B12, 21 stable GLCM)
* ``supervoxel``     -> ``params_supervoxel_radiomics.yaml`` (full texture classes)
* ``roi``            -> ``parameter.yaml`` (generic full set incl. shape)
* ``habitat``        -> ``parameter_habitat.yaml`` (habitat-map oriented)
"""
