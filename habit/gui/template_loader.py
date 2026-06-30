# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""Load habitat wizard JSON templates from ``habit/gui/templates/habitat/``."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TEMPLATE_DIR: Path = Path(__file__).resolve().parent / "templates" / "habitat"


@lru_cache(maxsize=1)
def _load_all_templates() -> Dict[str, Dict[str, Any]]:
    """
    Read all ``*.json`` templates under the habitat template directory.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of template id to template document.
    """
    templates: Dict[str, Dict[str, Any]] = {}
    if not _TEMPLATE_DIR.is_dir():
        return templates
    for path in sorted(_TEMPLATE_DIR.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        template_id = str(doc.get("id") or path.stem)
        doc["id"] = template_id
        templates[template_id] = doc
    return templates


def list_habitat_templates() -> List[Dict[str, Any]]:
    """
    Return all habitat wizard templates sorted by display name.

    Returns:
        List[Dict[str, Any]]: Template documents.
    """
    docs = list(_load_all_templates().values())
    docs.sort(key=lambda item: str(item.get("display_name", item.get("id", ""))))
    return docs


def load_habitat_template(template_id: str) -> Optional[Dict[str, Any]]:
    """
    Load one template document by id.

    Args:
        template_id: Template key such as ``liver_dce_two_step``.

    Returns:
        Optional[Dict[str, Any]]: Template document or None when missing.
    """
    if not template_id:
        return None
    return _load_all_templates().get(str(template_id))


def template_choices() -> List[Tuple[str, str]]:
    """
    Build Gradio Radio/Dropdown choices for templates.

    Returns:
        List[Tuple[str, str]]: ``(label, value)`` pairs for Gradio components.
    """
    choices: List[Tuple[str, str]] = []
    for doc in list_habitat_templates():
        template_id = str(doc.get("id", ""))
        label = str(doc.get("display_name", template_id))
        choices.append((label, template_id))
    return choices


def template_description(doc: Optional[Dict[str, Any]]) -> str:
    """
    Render a short Markdown description for a template document.

    Args:
        doc: Template JSON document.

    Returns:
        str: Markdown snippet for ``gr.Markdown``.
    """
    if not doc:
        return "_Template not found._"
    name = str(doc.get("display_name", doc.get("id", "Template")))
    desc = str(doc.get("description", "")).strip()
    modalities = doc.get("expected_modalities") or []
    mod_text = ", ".join(str(m) for m in modalities) if modalities else "—"
    clustering = str(doc.get("clustering_mode", "two_step"))
    prep = str(doc.get("prep_preset", "standard"))
    lines = [f"**{name}**"]
    if desc:
        lines.append(desc)
    lines.append(f"- Expected modalities: `{mod_text}`")
    lines.append(f"- Clustering: `{clustering}` · Preprocessing preset: `{prep}`")
    return "\n\n".join(lines)


__all__ = [
    "list_habitat_templates",
    "load_habitat_template",
    "template_choices",
    "template_description",
]
