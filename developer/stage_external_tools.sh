#!/usr/bin/env bash
# Stage portable-pack assets into the conda env BEFORE conda pack (macOS / Linux).
#
#   conda activate habit
#   bash developer/stage_external_tools.sh
#
# Pack root ($CONDA_PREFIX):
#   setup_habit.sh, install_gpu_torch.sh, requirements-gpu-torch-only.txt
# bin/:
#   optional dcm2niix, elastix, transformix from demo_data/linux or demo_data/darwin
#   (or install into the env first: conda install -c conda-forge dcm2niix)

set -euo pipefail

resolve_pack_root() {
    if [[ -n "${1:-}" ]]; then
        echo "$1"
        return
    fi
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        echo "$CONDA_PREFIX"
        return
    fi
    echo "[HABIT] ERROR: conda activate habit first, or pass env root:" >&2
    echo "        bash stage_external_tools.sh /path/to/env" >&2
    exit 1
}

PACK_ROOT="$(resolve_pack_root "${1:-}")"
BIN="${PACK_ROOT}/bin"

if [[ ! -x "${PACK_ROOT}/bin/python" && ! -f "${PACK_ROOT}/bin/python" ]]; then
    echo "[HABIT] ERROR: python not found in ${PACK_ROOT}/bin" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACK_SRC="${SCRIPT_DIR}"

case "$(uname -s)" in
    Linux)  TOOL_SRC="${REPO_ROOT}/demo_data/linux" ;;
    Darwin) TOOL_SRC="${REPO_ROOT}/demo_data/darwin" ;;
    *)      TOOL_SRC="" ;;
esac

echo
echo "=== Stage conda-pack assets (Unix) ==="
echo "Pack root:   ${PACK_ROOT}"
echo "Pack source: ${PACK_SRC}"
echo "Tool source: ${TOOL_SRC:-<none; use conda-installed tools in bin/>}"
echo

for f in setup_habit.sh install_gpu_torch.sh requirements-gpu-torch-only.txt; do
    if [[ ! -f "${PACK_SRC}/${f}" ]]; then
        echo "[HABIT] ERROR: Missing ${PACK_SRC}/${f}" >&2
        exit 1
    fi
done

echo "[1/2] Copy to pack root ..."
for f in setup_habit.sh install_gpu_torch.sh requirements-gpu-torch-only.txt; do
    cp -f "${PACK_SRC}/${f}" "${PACK_ROOT}/${f}"
    chmod +x "${PACK_ROOT}/${f}" 2>/dev/null || true
    echo "      ${f}"
done

echo "[2/2] Copy external tools to bin/ (optional) ..."
if [[ -n "${TOOL_SRC}" && -d "${TOOL_SRC}" ]]; then
    for f in dcm2niix elastix transformix; do
        if [[ -f "${TOOL_SRC}/${f}" ]]; then
            cp -f "${TOOL_SRC}/${f}" "${BIN}/${f}"
            chmod +x "${BIN}/${f}"
            echo "      bin/${f}"
        fi
    done
else
    echo "      Skipped (no demo_data/linux or demo_data/darwin)."
    echo "      Tip: conda install -c conda-forge dcm2niix into this env before pack,"
    echo "           or add binaries under demo_data/linux/ or demo_data/darwin/."
fi

echo
echo "Staging complete."
