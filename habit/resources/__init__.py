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
Bundled, importable resource data shipped inside the HABIT package.

* ``habit/resources/radiomics`` — default PyRadiomics parameter presets so
  ``params_file`` can be omitted (see :mod:`habit.utils.radiomics_preset_utils`).
* ``habit/resources/demo_config`` — build-time mirror of repository ``config/``
  for wheel installs (``habit.copy_demo_config`` / ``habit copy-demo-config``).
  Editable checkouts read ``config/`` directly; edit that tree only.
  ``demo_data/`` is never packaged.
"""
