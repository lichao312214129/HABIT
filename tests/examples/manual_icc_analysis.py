#!/usr/bin/env python
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
Manual example: ICC reliability via ``habit.evaluation.statistics.icc_analysis``.

Pure Python API — FeatureTable loaded from CSV, no YAML config.

Run from repository root (py310)::

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_icc_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from habit.contracts.table import FeatureTable
from habit.evaluation.statistics import icc_analysis

from tests.examples.demo_paths import EXAMPLE_OUT, ML_DATA

OUT_DIR: Path = EXAMPLE_OUT / "icc"
TEST_CSV: Path = ML_DATA / "breast_cancer_dataset.csv"
RETEST_CSV: Path = ML_DATA / "breast_cancer_dataset_retest_simulated.csv"
OUTPUT_JSON: Path = OUT_DIR / "icc_radiomics.json"

ID_COL: str = "subject_id"
EXCLUDE_COLS: frozenset[str] = frozenset({ID_COL, "label"})


def load_feature_table(csv_path: Path) -> FeatureTable:
    """Load a tabular feature CSV as a FeatureTable (features only, no outcome)."""
    frame: pd.DataFrame = pd.read_csv(csv_path, dtype={ID_COL: str})
    feature_columns: tuple[str, ...] = tuple(
        col for col in frame.columns if col not in EXCLUDE_COLS
    )
    return FeatureTable(
        frame=frame,
        id_columns=(ID_COL,),
        feature_columns=feature_columns,
    )


test_table: FeatureTable = load_feature_table(TEST_CSV)
retest_table: FeatureTable = load_feature_table(RETEST_CSV)

# Align on subjects present in both sessions (retest demo has 50 paired rows).
common_ids: list[str] = sorted(
    set(retest_table.frame[ID_COL].astype(str))
    & set(test_table.frame[ID_COL].astype(str))
)
test_frame: pd.DataFrame = test_table.frame.set_index(ID_COL).loc[common_ids].reset_index()
retest_frame: pd.DataFrame = retest_table.frame.set_index(ID_COL).loc[common_ids].reset_index()
feature_columns: tuple[str, ...] = test_table.feature_columns
test_table = FeatureTable(
    frame=test_frame,
    id_columns=(ID_COL,),
    feature_columns=feature_columns,
)
retest_table = FeatureTable(
    frame=retest_frame,
    id_columns=(ID_COL,),
    feature_columns=feature_columns,
)
print(f"Test CSV:   {TEST_CSV}  ({len(feature_columns)} features, {len(common_ids)} paired subjects)")
print(f"Retest CSV: {RETEST_CSV}")

icc_df = icc_analysis(test_table, [retest_table], icc_types=("icc2", "icc3"), verbose=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(
    json.dumps(icc_df.to_dict(orient="records"), indent=2),
    encoding="utf-8",
)
print(f"Saved ICC results: {OUTPUT_JSON}  ({len(icc_df)} features)")
print("Done.")
