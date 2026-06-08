#!/usr/bin/env bash
# Upgrade CPU torch to GPU torch inside a conda-pack HABIT environment (Linux + NVIDIA).
# macOS: GPU torch via pip is limited; prefer CPU pack or install from PyTorch docs.
#
# Usage (from pack root):
#   bash install_gpu_torch.sh
#   bash install_gpu_torch.sh wheel

set -euo pipefail

resolve_pack_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${script_dir}/bin/python" || -x "${script_dir}/bin/python" ]]; then
        echo "${script_dir}"
        return
    fi
    if [[ -f "${PWD}/bin/python" || -x "${PWD}/bin/python" ]]; then
        echo "${PWD}"
        return
    fi
    echo ""
}

PACK_ROOT="$(resolve_pack_root)"
PYTHON="${PACK_ROOT}/bin/python"
REQ_FILE="${PACK_ROOT}/requirements-gpu-torch-only.txt"

if [[ -z "${PACK_ROOT}" ]]; then
    echo "[HABIT] Cannot find pack root. Run from extracted pack directory." >&2
    exit 1
fi

echo
echo "=== HABIT GPU torch installer (Unix) ==="
echo "Pack root: ${PACK_ROOT}"
echo

"${PYTHON}" -c "import torch; print('  version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())" 2>/dev/null || {
    echo "[HABIT] torch is not installed in this environment." >&2
    exit 1
}

echo

if [[ "${1:-}" == "wheel" ]]; then
    WHEEL="${PACK_ROOT}/torch-2.4.0+cu121-cp310-cp310-linux_x86_64.whl"
    if [[ ! -f "${WHEEL}" ]]; then
        echo "[HABIT] Wheel not found: ${WHEEL}" >&2
        exit 1
    fi
    echo "Installing GPU torch from local wheel ..."
    "${PYTHON}" -m pip install --upgrade "${WHEEL}"
else
    if [[ ! -f "${REQ_FILE}" ]]; then
        echo "[HABIT] Missing ${REQ_FILE}" >&2
        exit 1
    fi
    echo "Installing GPU torch from requirements-gpu-torch-only.txt (~2 GB download) ..."
    "${PYTHON}" -m pip install --upgrade -r "${REQ_FILE}"
fi

echo
echo "After install:"
"${PYTHON}" -c "import torch; print('  version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
echo
echo "Done."
