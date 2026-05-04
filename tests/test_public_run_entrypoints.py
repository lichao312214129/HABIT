"""Public entry-point tests for workflow-style objects.

These tests intentionally inspect source files instead of importing HABIT
modules. Importing ``habit.core`` can load optional medical-imaging
dependencies such as radiomics, while the API naming contract only needs
static validation.
"""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _class_method_names(relative_path: str, class_name: str) -> set[str]:
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


def _class_names(relative_path: str) -> set[str]:
    """Return every top-level class name defined in a source file."""
    source_path = PROJECT_ROOT / relative_path
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    return {node.name for node in module.body if isinstance(node, ast.ClassDef)}


def test_preprocessing_batch_processor_uses_run_entrypoint() -> None:
    """BatchProcessor should expose only the unified workflow entry point."""
    method_names = _class_method_names(
        "habit/core/preprocessing/image_processor_pipeline.py",
        "BatchProcessor",
    )

    assert "run" in method_names
    assert "process_batch" not in method_names


def test_machine_learning_workflows_use_run_entrypoint() -> None:
    """ML workflows should implement the unified ``run`` entry point.

    Both the new canonical names (``HoldoutWorkflow`` / ``KFoldWorkflow``)
    and the deprecated aliases (``MachineLearningWorkflow`` /
    ``MachineLearningKFoldWorkflow``) must be present in source so external
    scripts importing the legacy names continue to work.
    """
    base_methods = _class_method_names(
        "habit/core/machine_learning/workflows/base.py",
        "BaseWorkflow",
    )
    holdout_path = "habit/core/machine_learning/workflows/holdout_workflow.py"
    kfold_path = "habit/core/machine_learning/workflows/kfold_workflow.py"

    holdout_classes = _class_names(holdout_path)
    kfold_classes = _class_names(kfold_path)

    assert "run" in base_methods
    assert "run_pipeline" not in base_methods

    # New canonical names are the source of truth.
    assert "HoldoutWorkflow" in holdout_classes
    assert "KFoldWorkflow" in kfold_classes
    holdout_methods = _class_method_names(holdout_path, "HoldoutWorkflow")
    kfold_methods = _class_method_names(kfold_path, "KFoldWorkflow")
    assert "run" in holdout_methods
    assert "run_pipeline" not in holdout_methods
    assert "run" in kfold_methods
    assert "run_pipeline" not in kfold_methods

    # Deprecated aliases must still resolve (kept as deprecation subclasses).
    assert "MachineLearningWorkflow" in holdout_classes
    assert "MachineLearningKFoldWorkflow" in kfold_classes
    legacy_holdout_methods = _class_method_names(
        holdout_path, "MachineLearningWorkflow"
    )
    legacy_kfold_methods = _class_method_names(
        kfold_path, "MachineLearningKFoldWorkflow"
    )
    assert "run" in legacy_holdout_methods
    assert "run" in legacy_kfold_methods
