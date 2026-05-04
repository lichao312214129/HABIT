"""Static contract tests for the K-Fold machine-learning result hand-off."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse(relative_path: str) -> ast.Module:
    """Parse a project source file without importing optional ML dependencies."""
    source_path = PROJECT_ROOT / relative_path
    return ast.parse(source_path.read_text(encoding="utf-8"))


def _class_node(module: ast.Module, class_name: str) -> ast.ClassDef:
    """Return a class node from a parsed module."""
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Class {class_name!r} not found")


def test_kfold_result_contract_exists() -> None:
    """K-Fold should have a structured result object beside holdout results."""
    module = _parse("habit/core/machine_learning/core/results.py")
    class_names = {node.name for node in module.body if isinstance(node, ast.ClassDef)}
    result_class = _class_node(module, "KFoldRunResult")
    method_names = {
        item.name
        for item in result_class.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    # Symmetric per-model objects must coexist with the workflow-level result.
    assert {
        "KFoldRunResult",
        "KFoldModelResult",
        "AggregatedModelResult",
    }.issubset(class_names)
    assert "create" in method_names
    assert "to_legacy_results" in method_names


def test_kfold_runner_returns_structured_result() -> None:
    """KFoldRunner should match the runner layer and return KFoldRunResult."""
    module = _parse("habit/core/machine_learning/runners/kfold.py")
    runner_class = _class_node(module, "KFoldRunner")
    base_names = {
        base.id
        for base in runner_class.bases
        if isinstance(base, ast.Name)
    }
    run_method = next(
        item
        for item in runner_class.body
        if isinstance(item, ast.FunctionDef) and item.name == "run"
    )

    assert "BaseRunner" in base_names
    assert isinstance(run_method.returns, ast.Name)
    assert run_method.returns.id == "KFoldRunResult"


def test_kfold_workflow_uses_reporting_components() -> None:
    """K-Fold workflow should delegate persistence to the reporting layer."""
    source_path = PROJECT_ROOT / "habit/core/machine_learning/workflows/kfold_workflow.py"
    source = source_path.read_text(encoding="utf-8")

    # The runner result must be exposed via the structured object.
    assert "self._run_result = self.runner.run(X=X, y=y)" in source
    assert "self.results = self._run_result.to_legacy_results()" in source
    assert "self.fold_pipelines = self._run_result.fold_pipelines" in source

    # Reporting components must be invoked instead of inline IO.
    assert "ModelStore(" in source
    assert "ReportWriter(" in source
    assert "PlotComposer(" in source
    assert "save_kfold_ensembles" in source
