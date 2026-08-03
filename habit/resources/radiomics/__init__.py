# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
