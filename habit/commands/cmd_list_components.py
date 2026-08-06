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
"""Implementation of the ``habit list-components`` command."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import click


def run_list_components(domain: Optional[str], as_json: bool) -> None:
    """
    List the registered components of one or all plugin domains.

    This is the CLI view of :func:`habit.api.plugins.list_plugins`: every
    name a YAML ``name:`` field or an expression node may reference, plus
    the implementation backing it.

    Args:
        domain: Restrict the listing to one plugin domain (e.g.
            ``combiner``); ``None`` lists every registry-backed domain.
        as_json: Emit machine-readable JSON instead of a text table.
    """
    from habit.api.plugins import list_plugins

    infos = list_plugins(domain)
    if as_json:
        payload: List[Dict[str, Any]] = [
            {
                "domain": info.domain,
                "name": info.name,
                "implementation": info.implementation,
                "params_schema": info.params_schema,
                "provider": info.provider,
            }
            for info in infos
        ]
        click.echo(json.dumps(payload, indent=2))
        return

    current_domain: Optional[str] = None
    for info in infos:
        if info.domain != current_domain:
            current_domain = info.domain
            click.echo(f"\n[{current_domain}]")
        click.echo(f"  {info.name}  ({info.implementation})")
