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
"""Demo data path constants for ``tests/examples/`` manual scripts."""

from __future__ import annotations

from pathlib import Path

# Repository root: tests/examples/demo_paths.py -> parents[2]
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEMO_DATA: Path = REPO_ROOT / "demo_data"
IMAGING_ROOT: Path = DEMO_DATA / "preprocessed" / "processed_images"
ML_DATA: Path = DEMO_DATA / "ml_data"
EXAMPLE_OUT: Path = DEMO_DATA / "results" / "examples"
DICOM_ROOT: Path = DEMO_DATA / "dicom"
DCM2NIIX: Path = REPO_ROOT / "tools" / "bin" / "dcm2niix.exe"

# DCE-MRI modality keys under IMAGING_ROOT/images/<subject>/
MODALITIES: tuple[str, ...] = ("delay2", "delay3", "delay5")
