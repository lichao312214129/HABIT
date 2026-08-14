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
"""Stable HABIT plugin discovery and inspection API."""

from habit.api.plugins import (
    PluginCatalogEntry,
    PluginInfo,
    PluginLoadReport,
    PluginParamInfo,
    format_plugin_catalog_rst,
    get_param_schema,
    get_plugin_info,
    list_plugins,
    load_plugins,
    plugin_catalog,
)

__all__ = [
    "PluginInfo",
    "PluginParamInfo",
    "PluginCatalogEntry",
    "PluginLoadReport",
    "list_plugins",
    "get_plugin_info",
    "get_param_schema",
    "plugin_catalog",
    "format_plugin_catalog_rst",
    "load_plugins",
]
