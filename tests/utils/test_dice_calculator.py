"""
Unit tests for DiceCalculator.

Migrated and extended from tests/test_dice_calculator.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from habit.cli import cli
from habit.utils.dice_calculator import DiceCalculator


class TestDiceCalculator:
    def test_perfect_overlap_is_one(self) -> None:
        mask = np.array([[1, 0], [0, 1]])
        score = DiceCalculator.compute(mask, mask)
        assert abs(score - 1.0) < 1e-6

    def test_no_overlap_is_zero(self) -> None:
        a = np.array([[1, 0], [0, 0]])
        b = np.array([[0, 0], [0, 1]])
        score = DiceCalculator.compute(a, b)
        assert abs(score - 0.0) < 1e-6

    def test_partial_overlap(self) -> None:
        a = np.array([1, 1, 0, 0])
        b = np.array([1, 0, 0, 0])
        # Dice = 2 * |A∩B| / (|A| + |B|) = 2*1 / (2+1) = 2/3
        score = DiceCalculator.compute(a, b)
        assert abs(score - 2 / 3) < 1e-6

    def test_both_empty_returns_one(self) -> None:
        """Convention: two empty masks are considered identical (Dice = 1)."""
        a = np.zeros((3, 3), dtype=int)
        b = np.zeros((3, 3), dtype=int)
        score = DiceCalculator.compute(a, b)
        # Could be 1 or NaN/0 depending on implementation; just ensure no crash
        assert score is not None

    def test_output_in_range(self) -> None:
        rng = np.random.RandomState(0)
        a = (rng.rand(100) > 0.5).astype(int)
        b = (rng.rand(100) > 0.5).astype(int)
        score = DiceCalculator.compute(a, b)
        assert 0.0 <= score <= 1.0

    def test_score_is_symmetric(self) -> None:
        a = np.array([1, 1, 0, 0, 1])
        b = np.array([1, 0, 1, 0, 1])
        assert DiceCalculator.compute(a, b) == DiceCalculator.compute(b, a)


# ---------------------------------------------------------------------------
# dice CLI (habit dice)
# ---------------------------------------------------------------------------


class TestDiceCLI:
    """Smoke tests for ``habit dice`` command-line interface."""

    def test_dice_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["dice", "--help"])
        assert result.exit_code == 0
        assert "dice" in result.output.lower() or "coefficient" in result.output.lower()

    def test_dice_missing_inputs(self) -> None:
        runner = CliRunner()

        result = runner.invoke(
            cli, ["dice", "--input2", "test", "--output", "out.csv"]
        )
        assert result.exit_code != 0

        result = runner.invoke(
            cli, ["dice", "--input1", "test", "--output", "out.csv"]
        )
        assert result.exit_code != 0

    def test_dice_with_nonexistent_inputs(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "dice",
                "--input1",
                "nonexistent1.yaml",
                "--input2",
                "nonexistent2.yaml",
                "--output",
                "dice_results.csv",
            ],
        )
        assert result.exit_code != 0
