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
"""二步法生境 — 原始强度拼接特征 — 预测

Config: config/habitat/config_habitat_two_step_predict.yaml
Run:    python tests/habitat/habitat_two_step_raw_concat_predict.py

Edit the YAML above (#%% path blocks) for your own data. Optional: pass --debug
"""

import os
import sys
from pathlib import Path


def main() -> None:
    """Invoke habit CLI from repository root (Windows spawn-safe)."""
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = ["habit", "get-habitat", "-c", "config/habitat/config_habitat_two_step_predict.yaml", *sys.argv[1:]]
    from habit.cli import cli

    cli()


if __name__ == "__main__":
    main()
