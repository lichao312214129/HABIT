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
"""Safe arithmetic voxel features from modality intensities.

This covers the common DIY case that ``raw`` / ``concat`` cannot express:
per-voxel formulas that combine modalities (ratios, powers, logs, ...).
Expressions are evaluated through a restricted AST -- not ``eval`` -- so
arbitrary Python cannot escape into the process.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field, model_validator, ConfigDict

from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.subject import Subject
from habit.domain.voxel_features._base import (
    aligned_image,
    build_voxel_field,
    roi_voxels,
)
from habit.domain.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec

__all__ = [
    "ExpressionVoxelFeatures",
    "ExpressionVoxelFeaturesParams",
    "normalize_expression_source",
]

#: Names injected into every expression namespace (not read from the subject).
_BUILTINS: Mapping[str, Any] = {
    "eps": 1e-8,
    "pi": math.pi,
    "e": math.e,
}

#: Callables allowed inside expressions. Each maps arrays element-wise.
_FUNCTIONS: Mapping[str, Any] = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "square": np.square,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
}


class ExpressionVoxelFeaturesParams(BaseModel):
    """Constructor parameters for :class:`ExpressionVoxelFeatures`."""

    model_config = ConfigDict(extra="forbid")
    features: Optional[Dict[str, str]] = None
    expressions: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    modalities: List[str] = Field(default_factory=list)
    roi: Optional[str] = None
    eps: float = 1e-8

    @model_validator(mode="after")
    def _require_features_or_expressions(self) -> "ExpressionVoxelFeaturesParams":
        """Accept either a name->formula map or a parallel expression list."""
        if self.features and self.expressions:
            raise ValueError(
                "Provide either 'features' or 'expressions', not both."
            )
        if not self.features and not self.expressions:
            raise ValueError(
                "ExpressionVoxelFeatures requires 'features' (name->formula) "
                "or 'expressions' (ordered formulas)."
            )
        return self


def normalize_expression_source(source: str) -> str:
    """
    Normalise user-facing syntax before AST parsing.

    Science users often write ``^`` for power; Python's AST treats ``^`` as
    bitwise XOR, which is almost never intended for intensities. Replace
    ``^`` with ``**`` before parsing. Whitespace is left alone.

    Args:
        source: Raw expression text from the Spec.

    Returns:
        Expression text safe to feed to :func:`ast.parse`.
    """
    return source.replace("^", "**")


def _collect_names(tree: ast.AST) -> Tuple[str, ...]:
    """Return identifier names referenced in ``tree``, sorted uniquely."""
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    return tuple(sorted(names))


class _SafeEvaluator(ast.NodeVisitor):
    """
    Evaluate a restricted expression AST against a name->array namespace.

    Allowed nodes: constants, names, unary ``+/-``, binary
    ``+ - * / // % **``, and calls to the whitelisted functions. Everything
    else raises :class:`HABITAPIError`.
    """

    def __init__(self, namespace: Mapping[str, Any]) -> None:
        self._namespace = namespace

    def visit(self, node: ast.AST) -> Any:
        """Dispatch with a clear error for unsupported syntax."""
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise HABITAPIError(
                f"expression: unsupported syntax {type(node).__name__!r}."
            )
        return method(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        """Evaluate the body of an ``ast.parse(..., mode='eval')`` tree."""
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Allow numeric literals only."""
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise HABITAPIError(
                f"expression: only numeric literals are allowed, got {node.value!r}."
            )
        return float(node.value)

    def visit_Name(self, node: ast.Name) -> Any:
        """Resolve a modality, constant, or injected name."""
        if node.id not in self._namespace:
            raise HABITAPIError(
                f"expression: unknown name {node.id!r}; available: "
                f"{sorted(self._namespace)}."
            )
        return self._namespace[node.id]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Evaluate unary plus/minus."""
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise HABITAPIError(
            f"expression: unsupported unary operator {type(node.op).__name__}."
        )

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Evaluate a binary arithmetic operator."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left ** right
        raise HABITAPIError(
            f"expression: unsupported binary operator {type(op).__name__}."
        )

    def visit_Call(self, node: ast.Call) -> Any:
        """Evaluate a whitelisted element-wise function call."""
        if not isinstance(node.func, ast.Name):
            raise HABITAPIError(
                "expression: only bare function names may be called "
                f"(got {ast.dump(node.func)})."
            )
        if node.keywords:
            raise HABITAPIError(
                "expression: keyword arguments are not allowed in function calls."
            )
        name = node.func.id
        if name not in _FUNCTIONS:
            raise HABITAPIError(
                f"expression: unknown function {name!r}; "
                f"allowed: {sorted(_FUNCTIONS)}."
            )
        args = [self.visit(arg) for arg in node.args]
        try:
            return _FUNCTIONS[name](*args)
        except TypeError as exc:
            raise HABITAPIError(
                f"expression: bad arguments for {name}(): {exc}"
            ) from exc


#: AST node types the evaluator knows how to visit.
_ALLOWED_NODES: Tuple[type, ...] = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.Call,
)


def _compile_expression(source: str) -> ast.Expression:
    """Parse and validate one expression string."""
    text = normalize_expression_source(source).strip()
    if not text:
        raise HABITAPIError("expression: empty formula.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise HABITAPIError(
            f"expression: cannot parse {source!r}: {exc.msg}"
        ) from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise HABITAPIError(
                f"expression: unsupported syntax {type(node).__name__!r} "
                f"in {source!r}."
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise HABITAPIError(
                    f"expression: only bare function names may be called "
                    f"in {source!r}."
                )
            if node.func.id not in _FUNCTIONS:
                raise HABITAPIError(
                    f"expression: unknown function {node.func.id!r}; "
                    f"allowed: {sorted(_FUNCTIONS)}."
                )
            if node.keywords:
                raise HABITAPIError(
                    "expression: keyword arguments are not allowed in "
                    f"function calls ({source!r})."
                )
    return tree  # type: ignore[return-value]


def _resolve_feature_table(
    features: Optional[Mapping[str, str]],
    expressions: Optional[Sequence[str]],
    feature_names: Optional[Sequence[str]],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Normalise constructor inputs into parallel (names, formulas) tuples.

    Args:
        features: Optional name->formula mapping.
        expressions: Optional ordered formulas.
        feature_names: Optional names aligned with ``expressions``.

    Returns:
        ``(names, formulas)`` of equal length.
    """
    if features:
        names = tuple(str(key) for key in features.keys())
        formulas = tuple(str(value) for value in features.values())
        return names, formulas
    assert expressions is not None
    formulas = tuple(str(item) for item in expressions)
    if not formulas:
        raise HABITAPIError("expression: 'expressions' must not be empty.")
    if feature_names is not None:
        names = tuple(str(item) for item in feature_names)
        if len(names) != len(formulas):
            raise HABITAPIError(
                f"expression: got {len(feature_names)} feature_names for "
                f"{len(formulas)} expressions."
            )
        return names, formulas
    return tuple(f"expr_{index}" for index in range(len(formulas))), formulas


@VoxelFeatureExtractorRegistry.register("expression")
class ExpressionVoxelFeatures:
    """
    Per-voxel features defined by restricted arithmetic expressions.

    Each formula may reference modality keys on the subject (e.g. ``T1``,
    ``T2``), the injected constant ``eps`` (override via the constructor),
    and the whitelisted functions ``abs``, ``sqrt``, ``square``, ``log``,
    ``log10``, ``exp``, ``minimum``, ``maximum``, ``clip``. Power may be
    written as ``**`` or ``^``.

    Example::

        Spec("expression", {
            "features": {
                "t1_over_t2_sq": "square(T1 / (T2 ** 3 + eps))",
            },
        })

    Args:
        features: Mapping of feature name to formula. Mutually exclusive
            with ``expressions``.
        expressions: Ordered formulas when names are not provided up front.
        feature_names: Names aligned with ``expressions``; defaults to
            ``expr_0``, ``expr_1``, ...
        modalities: Optional explicit modality list. When empty, modality
            names are inferred from identifiers used in the formulas
            (excluding builtins and function names).
        roi: Mask key; ``None`` uses the subject's single mask.
        eps: Value bound to the name ``eps`` inside every formula.
    """

    def __init__(
        self,
        features: Optional[Mapping[str, str]] = None,
        expressions: Optional[Sequence[str]] = None,
        feature_names: Optional[Sequence[str]] = None,
        modalities: Sequence[str] = (),
        roi: Optional[str] = None,
        eps: float = 1e-8,
    ) -> None:
        names, formulas = _resolve_feature_table(features, expressions, feature_names)
        self.feature_names: Tuple[str, ...] = names
        self.formulas: Tuple[str, ...] = formulas
        self.trees: Tuple[ast.Expression, ...] = tuple(
            _compile_expression(formula) for formula in formulas
        )
        self.roi = roi
        self.eps = float(eps)
        inferred = self._infer_modalities()
        if modalities:
            self.modalities: Tuple[str, ...] = tuple(str(name) for name in modalities)
            missing = [name for name in inferred if name not in self.modalities]
            if missing:
                raise HABITAPIError(
                    "expression: formulas reference modalities "
                    f"{missing} that are not listed in modalities={list(self.modalities)}."
                )
        else:
            self.modalities = inferred
        if not self.modalities:
            raise HABITAPIError(
                "expression: no modality names found in formulas; reference "
                "at least one subject modality (e.g. 'T1')."
            )

    def _infer_modalities(self) -> Tuple[str, ...]:
        """Collect modality identifiers referenced by any compiled formula."""
        reserved = set(_BUILTINS) | set(_FUNCTIONS) | {"eps"}
        names: List[str] = []
        seen = set()
        for tree in self.trees:
            for name in _collect_names(tree):
                if name in reserved or name in seen:
                    continue
                seen.add(name)
                names.append(name)
        return tuple(names)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="expression",
            params={
                "features": dict(zip(self.feature_names, self.formulas)),
                "modalities": list(self.modalities),
                "roi": self.roi,
                "eps": self.eps,
            },
        )

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """
        Evaluate every formula on the ROI voxels of ``subject``.

        Args:
            subject: Subject providing the referenced modalities and mask.

        Returns:
            One column per formula, one row per ROI voxel.

        Raises:
            KeyError: If a referenced modality or the ROI is absent.
            GeometryError: If a modality and the mask do not share a grid.
            HABITAPIError: If a formula uses unsupported syntax or names.
        """
        mask, inside, voxel_index = roi_voxels(subject, self.roi)
        namespace: Dict[str, Any] = {
            **_BUILTINS,
            "eps": self.eps,
        }
        for modality in self.modalities:
            array = aligned_image(subject, modality, mask, owner="expression")
            namespace[modality] = np.asarray(array[inside], dtype=np.float64)

        columns: List[np.ndarray] = []
        for name, tree in zip(self.feature_names, self.trees):
            value = _SafeEvaluator(namespace).visit(tree)
            column = np.asarray(value, dtype=np.float64)
            if column.shape != (inside.sum(),):
                # Broadcast scalars (e.g. literal-only formulas) to every voxel.
                if column.ndim == 0:
                    column = np.full(int(inside.sum()), float(column), dtype=np.float64)
                else:
                    raise HABITAPIError(
                        f"expression: formula for {name!r} produced shape "
                        f"{column.shape}, expected {(int(inside.sum()),)}."
                    )
            columns.append(column)

        values = np.stack(columns, axis=1)
        return build_voxel_field(
            subject, mask, voxel_index, self.feature_names, values, self.spec
        )


VoxelFeatureExtractorRegistry.register_params_model(
    "expression", ExpressionVoxelFeaturesParams
)
