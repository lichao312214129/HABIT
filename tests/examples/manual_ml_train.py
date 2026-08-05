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
Manual example: hold-out model training via ``habit.recipes.train_model``.

Pure Python API — MLSpec built in code, FeatureTable loaded from CSV.

Run from repository root (py310)::

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_ml_train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from habit.contracts.outcome import BinaryOutcome
from habit.contracts.table import FeatureTable
from habit.recipes import train_model
from habit.spec.specs import MLSpec, Spec

from tests.examples.demo_paths import EXAMPLE_OUT, ML_DATA

OUT_DIR: Path = EXAMPLE_OUT / "ml_train"
CSV_PATH: Path = ML_DATA / "breast_cancer_dataset.csv"
ID_COL: str = "subject_id"
LABEL_COL: str = "label"
SEED: int = 42

# --- build MLSpec in code ---
spec = MLSpec(
    name="demo_ml_train",
    classifier=Spec(
        name="LogisticRegression",
        params={"max_iter": 1000, "C": 1.0, "penalty": "l2", "solver": "lbfgs"},
    ),
    table_preprocessors=(Spec(name="zscore"),),
    feature_selectors=(
        Spec(name="variance", params={"threshold": 0.2, "plot_variances": True}),
        Spec(name="correlation", params={"threshold": 0.8, "method": "spearman", "visualize": True}),
    ),
    random_seed=SEED,
)

# --- load FeatureTable from CSV ---
frame: pd.DataFrame = pd.read_csv(CSV_PATH, dtype={ID_COL: str})
feature_columns: tuple[str, ...] = tuple(
    col for col in frame.columns if col not in (ID_COL, LABEL_COL)
)
table = FeatureTable(
    frame=frame,
    id_columns=(ID_COL,),
    feature_columns=feature_columns,
    outcome=BinaryOutcome(column=LABEL_COL),
)
print(f"Feature table: {table.frame.shape[0]} rows, {len(table.feature_columns)} features")

# --- custom train/test split from id files ---
train_ids: list[str] = (ML_DATA / "train_ids.txt").read_text(encoding="utf-8").splitlines()
test_ids: list[str] = (ML_DATA / "test_ids.txt").read_text(encoding="utf-8").splitlines()

result = train_model(table, spec, seed=SEED, train_ids=train_ids, test_ids=test_ids)

print("Train metrics:")
for name, value in result.train_metrics.items():
    print(f"  {name}: {value:.4f}")
if result.test_metrics is not None:
    print("Held-out metrics:")
    for name, value in result.test_metrics.items():
        print(f"  {name}: {value:.4f}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
metrics_path: Path = OUT_DIR / "metrics.json"
metrics_path.write_text(
    json.dumps(
        {"train": dict(result.train_metrics), "test": dict(result.test_metrics) if result.test_metrics else None},
        indent=2,
    ),
    encoding="utf-8",
)
print(f"Saved summary metrics: {metrics_path}")
print("Done.")
