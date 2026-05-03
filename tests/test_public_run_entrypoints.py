"""Public entry-point tests for workflow-style objects.

These tests intentionally inspect source files instead of importing HABIT
modules. Importing ``habit.core`` can load optional medical-imaging dependencies
such as radiomics, while the API naming contract only needs static validation.
"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _method_names(relative_path: str, class_name: str) -> set[str]:
    """Return method names defined directly on a class in a source file."""
    source_path = PROJECT_ROOT / relative_path
    module = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

    raise AssertionError(f"Class {class_name!r} not found in {source_path}")


def test_preprocessing_batch_processor_uses_run_entrypoint() -> None:
    """BatchProcessor should expose only the unified workflow entry point."""
    method_names = _method_names(
        "habit/core/preprocessing/image_processor_pipeline.py",
        "BatchProcessor",
    )

    assert "run" in method_names
    assert "process_batch" not in method_names


def test_machine_learning_workflows_use_run_entrypoint() -> None:
    """ML workflows should implement the same public entry point as other modules."""
    base_methods = _method_names(
        "habit/core/machine_learning/base_workflow.py",
        "BaseWorkflow",
    )
    holdout_methods = _method_names(
        "habit/core/machine_learning/workflows/holdout_workflow.py",
        "MachineLearningWorkflow",
    )
    kfold_methods = _method_names(
        "habit/core/machine_learning/workflows/kfold_workflow.py",
        "MachineLearningKFoldWorkflow",
    )

    assert "run" in base_methods
    assert "run_pipeline" not in base_methods
    assert "run" in holdout_methods
    assert "run_pipeline" not in holdout_methods
    assert "run" in kfold_methods
    assert "run_pipeline" not in kfold_methods
