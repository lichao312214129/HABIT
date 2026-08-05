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
Reflect Pydantic param models into JSON field descriptors for GUI APIs.

Single export path used by ``habit-gui/bridge`` schema reflection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Type

from pydantic import BaseModel

from habit.schemas.field_reflect import FieldDescriptor, reflect_schema

# Runtime keys filled by the pipeline — never shown in user forms.
AUTO_FILLED_FIELD_NAMES: frozenset[str] = frozenset({"images", "before_z_score"})


def descriptor_to_json(desc: FieldDescriptor) -> Dict[str, Any]:
    """Serialize one :class:`FieldDescriptor` for REST / React consumers."""
    return {
        "name": desc.name,
        "kind": desc.kind,
        "default": desc.default,
        "description": desc.description,
        "required": desc.required,
        "choices": desc.choices,
        "ge": desc.ge,
        "gt": desc.gt,
        "le": desc.le,
        "lt": desc.lt,
        "min_length": desc.min_length,
        "max_length": desc.max_length,
        "is_list": desc.is_list,
        "is_optional": desc.is_optional,
        "is_path": desc.is_path,
        "path_type": desc.path_type,
        "group": desc.group or "General",
        "order": desc.order,
        "widget": desc.widget,
        "visible_if": desc.visible_if,
        "label": desc.label,
        "help_text": desc.help_text,
        "display_label": desc.display_label,
    }


def reflect_param_model(
    model_cls: Type[BaseModel],
    *,
    exclude: Set[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Reflect a registered *Params model to GUI-ready JSON field descriptors.

    Args:
        model_cls: Pydantic params schema for one registry step.
        exclude: Additional field names to omit from the form.

    Returns:
        List[Dict[str, Any]]: Serialized field descriptors.
    """
    exclude_set: Set[str] = set(exclude or set()) | set(AUTO_FILLED_FIELD_NAMES)
    descriptors: List[FieldDescriptor] = [
        desc for desc in reflect_schema(model_cls) if desc.name not in exclude_set
    ]
    return [descriptor_to_json(desc) for desc in descriptors]
