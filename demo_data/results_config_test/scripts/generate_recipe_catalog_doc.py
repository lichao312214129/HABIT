#!/usr/bin/env python3
"""Generate docs/source/configuration/recipe_catalog.rst from matrix_manifest + summary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Cannot locate repo root from {start}")


def _status_badge(status: Optional[str]) -> str:
    if status in ("passed", "passed_empty"):
        return "PASS"
    if status == "skipped":
        return "SKIP"
    if status == "failed":
        return "FAIL"
    if status == "dry_run":
        return "DRY-RUN"
    return "PENDING"


def _cli_command(command: str, base_config: str, extra_args: List[str]) -> str:
    args = " ".join(extra_args) if extra_args else ""
    return f"habit {command} -c {base_config}" + (f" {args}" if args else "")


def build_catalog(
    manifest: Dict[str, Any],
    summary: Optional[Dict[str, Any]],
) -> str:
    """Build reStructuredText catalog body."""
    summary_by_id: Dict[str, Dict[str, Any]] = {}
    if summary:
        for case in summary.get("cases", []):
            summary_by_id[case["case_id"]] = case

    finished = summary.get("finished_at", "not run") if summary else "not run"
    passed = summary.get("passed", "-") if summary else "-"
    failed = summary.get("failed", "-") if summary else "-"
    skipped = summary.get("skipped", "-") if summary else "-"

    lines: List[str] = [
        "Recipe Catalog (Coverage Matrix)",
        "================================",
        "",
        "This page lists **tested configuration recipes** for HABIT v1 CLI workflows.",
        "Each row maps to an entry in ``demo_data/results_config_test/matrix_manifest.yaml``.",
        "",
        f"Last matrix run: {finished} (passed={passed}, failed={failed}, skipped={skipped})",
        "",
        "Quick run (from repository root, ``habit`` conda env):",
        "",
        ".. code-block:: bash",
        "",
        "   python demo_data/results_config_test/scripts/run_config_matrix.py --include-slow",
        "",
        "Recipe YAML paths are relative to the repository root. Copy a template, edit",
        "``#%%====`` path blocks, then run the CLI command shown in the table.",
        "",
    ]

    cases: List[Dict[str, Any]] = manifest.get("cases", [])
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        cat = case.get("category", "other")
        by_category.setdefault(cat, []).append(case)

    category_titles = {
        "machine_learning": "Machine Learning",
        "habitat": "Habitat Segmentation",
        "feature_extraction": "Feature Extraction",
        "preprocessing": "Preprocessing",
        "model_comparison": "Model Comparison",
        "auxiliary": "Auxiliary Tools",
        "radiomics": "Traditional Radiomics",
        "other": "Other",
    }

    for cat in sorted(by_category.keys(), key=lambda c: list(category_titles.keys()).index(c)
                      if c in category_titles else 999):
        title = category_titles.get(cat, cat.replace("_", " ").title())
        lines.extend([title, "-" * len(title), ""])
        lines.extend([
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 18 22 28 8",
            "",
            "   * - Recipe ID",
            "     - Case",
            "     - CLI",
            "     - Status",
        ])
        for case in by_category[cat]:
            case_id = case["case_id"]
            recipe_id = case.get("recipe_id", case_id)
            name = case.get("name", case_id)
            cmd = _cli_command(
                case["command"],
                case["base_config"],
                case.get("extra_args") or [],
            )
            rec = summary_by_id.get(case_id, {})
            status = _status_badge(rec.get("status"))
            reason = rec.get("reason", "")
            if status == "SKIP" and reason:
                status = f"SKIP ({reason[:40]}...)" if len(reason) > 40 else f"SKIP ({reason})"
            lines.append(f"   * - ``{recipe_id}``")
            lines.append(f"     - {name}")
            lines.append(f"     - ``{cmd}``")
            lines.append(f"     - {status}")
        lines.append("")

    lines.extend([
        "See Also",
        "--------",
        "",
        "- :doc:`preprocessing` — preprocessing parameter reference",
        "- :doc:`habitat` — habitat segmentation parameter reference",
        "- :doc:`feature_extraction` — feature extraction parameter reference",
        "- :doc:`machine_learning` — ML parameter reference",
        "- :doc:`auxiliary` — ICC, test-retest, and utility commands",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    script = Path(__file__).resolve()
    repo_root = _find_repo_root(script)
    results_root = repo_root / "demo_data" / "results_config_test"
    manifest_path = results_root / "matrix_manifest.yaml"
    summary_path = results_root / "matrix_summary.json"

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    summary: Optional[Dict[str, Any]] = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    body = build_catalog(manifest, summary)
    out_path = repo_root / "docs" / "source" / "configuration" / "recipe_catalog.rst"
    out_path.write_text(body, encoding="utf-8")
    print(f"Wrote {out_path} ({len(manifest.get('cases', []))} recipes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
