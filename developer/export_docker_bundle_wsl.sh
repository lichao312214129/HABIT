#!/usr/bin/env bash
# WSL/Linux entry: fix CRLF then run export.
#
# Use when:
#   - You are already inside Ubuntu/WSL (recommended for repos under ~/...)
#   - export_docker_bundle.bat fails before WSL delegation (rare)
#   - You see $'\r': command not found from CRLF line endings
#
# From PowerShell on \\wsl.localhost\... you may also run:
#   .\developer\export_docker_bundle.bat
# which auto-delegates to export_docker_bundle.sh inside WSL.
python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fix_sh_line_endings.py" developer docker
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/export_docker_bundle.sh" "$@"
