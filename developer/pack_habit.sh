#!/usr/bin/env bash
# Build HABIT portable pack (conda pack) — macOS / Linux maintainer entry point.
#
#   conda activate habit
#   bash developer/pack_habit.sh
#   bash developer/pack_habit.sh HABIT-linux-py310-gpu-v0.1.0.tar.gz

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "[HABIT] ERROR: conda activate habit first." >&2
    exit 1
fi

ENV_NAME="${CONDA_DEFAULT_ENV:-habit}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

default_output_name() {
    case "$(uname -s)" in
        Darwin) echo "HABIT-macos-py310-cpu-v0.1.0.tar.gz" ;;
        Linux)  echo "HABIT-linux-py310-gpu-v0.1.0.tar.gz" ;;
        *)      echo "HABIT-unix-py310-v0.1.0.tar.gz" ;;
    esac
}

OUTPUT="${1:-$(default_output_name)}"

resolve_conda_cmd() {
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        echo "${CONDA_EXE}"
        return
    fi
    if [[ -x "${CONDA_PREFIX}/bin/conda" ]]; then
        echo "${CONDA_PREFIX}/bin/conda"
        return
    fi
    if [[ -x "${CONDA_PREFIX}/../../bin/conda" ]]; then
        echo "$(cd "${CONDA_PREFIX}/../../bin" && pwd)/conda"
        return
    fi
    if command -v conda >/dev/null 2>&1; then
        echo "conda"
        return
    fi
    echo "[HABIT] ERROR: conda not found. Activate the env or set CONDA_EXE." >&2
    exit 1
}

CONDA_CMD="$(resolve_conda_cmd)"

echo
echo "=== HABIT conda pack build (Unix) ==="
echo "Environment: ${ENV_NAME}"
echo "Output:      ${OUTPUT}"
echo "Conda:       ${CONDA_CMD}"
echo

bash "${SCRIPT_DIR}/stage_external_tools.sh"

echo
echo "[3/3] Running conda pack ..."
"${CONDA_CMD}" pack -n "${ENV_NAME}" -o "${OUTPUT}" --compress-level 9

echo
echo "Done: ${OUTPUT}"
