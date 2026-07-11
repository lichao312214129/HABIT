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
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_EXE="${CONDA_PREFIX}/bin/python"
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/habit-pack-build.XXXXXX")"
trap 'rm -rf "${BUILD_ROOT}"' EXIT

if [[ ! -x "${PYTHON_EXE}" ]]; then
    echo "[HABIT] ERROR: Python not found at ${PYTHON_EXE}." >&2
    exit 1
fi

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

echo "[1/4] Building and force-reinstalling HABIT from the current checkout ..."
mkdir -p "${BUILD_ROOT}/build" "${BUILD_ROOT}/bdist" "${BUILD_ROOT}/dist"
(
    cd "${REPO_ROOT}"
    "${PYTHON_EXE}" setup.py \
        build --build-base "${BUILD_ROOT}/build" \
        bdist_wheel --bdist-dir "${BUILD_ROOT}/bdist" \
        --dist-dir "${BUILD_ROOT}/dist"
)
shopt -s nullglob
HABIT_WHEELS=("${BUILD_ROOT}"/dist/*.whl)
shopt -u nullglob
if [[ "${#HABIT_WHEELS[@]}" -ne 1 ]]; then
    echo "[HABIT] ERROR: Expected one HABIT wheel, found ${#HABIT_WHEELS[@]}." >&2
    exit 1
fi
"${PYTHON_EXE}" -m pip install --force-reinstall --no-deps "${HABIT_WHEELS[0]}"

echo
echo "[2/4] Validating the installed package from a neutral directory ..."
(
    cd "${TMPDIR:-/tmp}"
    "${PYTHON_EXE}" -s -c "import os, pathlib, habit; env = pathlib.Path(os.environ['CONDA_PREFIX']).resolve(); package = pathlib.Path(habit.__file__).resolve(); assert env in package.parents, f'HABIT loaded outside target environment: {package}'; from habit.core.machine_learning.evaluation.metrics import delong_roc_test; from habit.core.habitat_analysis.clustering_features.supervoxel_cext import is_cext_available; from habit.utils.radiomics_preset_utils import get_preset_path; assert is_cext_available(), 'HABIT native C extension is unavailable'; assert pathlib.Path(get_preset_path('voxel')).is_file(), 'Bundled radiomics preset is missing'; print(f'[HABIT] Installed package verified: {package}')"
)

echo
echo "[3/4] Staging portable tools ..."
bash "${SCRIPT_DIR}/stage_external_tools.sh"

echo
echo "[4/4] Running conda pack ..."
"${CONDA_CMD}" pack -n "${ENV_NAME}" -o "${OUTPUT}" --compress-level 9

echo
echo "Done: ${OUTPUT}"
