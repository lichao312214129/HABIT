#!/usr/bin/env bash
# Build HABIT Docker image and assemble an offline distribution bundle for end users.
#
#   bash developer/export_docker_bundle.sh
#   bash developer/export_docker_bundle.sh 0.1.0
#
# Windows entry points:
#   developer\export_docker_bundle.bat          — native drive paths (C:\, D:\, ...)
#   developer\export_docker_bundle.bat          — auto-delegates here when repo is
#                                                 opened via \\wsl.localhost\... or \\wsl$\
#   bash developer/export_docker_bundle_wsl.sh  — same as this script (CRLF fix wrapper)
#
# If you see $'\r': command not found, run once:
#   python developer/fix_sh_line_endings.py

# Normalize *.sh to LF before set -euo pipefail (safe no-op when already LF).
if command -v python3 >/dev/null 2>&1; then
    _habit_fix_py="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fix_sh_line_endings.py"
    if [[ -f "${_habit_fix_py}" ]]; then
        python3 "${_habit_fix_py}" developer docker >/dev/null 2>&1 || true
    fi
    unset _habit_fix_py
fi

set -euo pipefail

VERSION="${1:-0.1.0}"
IMAGE_TAG="habit:${VERSION}-cpu"
TAR_NAME="habit-${VERSION}-cpu.tar"
BUNDLE_NAME="HABIT-docker-v${VERSION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/docker"
OUT_DIR="${REPO_ROOT}/${BUNDLE_NAME}"

echo
echo "=== HABIT Docker bundle export v${VERSION} ==="
echo "Image tag: ${IMAGE_TAG}"
echo "Output:    ${OUT_DIR}"
echo

cd "${REPO_ROOT}"

echo "[1/4] Building Docker image (CPU) ..."
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t "${IMAGE_TAG}" .

echo "[2/4] Saving image to tar ..."
mkdir -p "${OUT_DIR}/images"
docker save "${IMAGE_TAG}" -o "${OUT_DIR}/images/${TAR_NAME}"

echo "[3/4] Copying launcher files ..."
mkdir -p "${OUT_DIR}/data" "${OUT_DIR}/config" "${OUT_DIR}/output"

for f in docker-compose.yml start-gui.bat stop-gui.bat start-gui.sh stop-gui.sh \
    generate-compose-override.ps1 generate-compose-override.sh ensure-docker-prereqs.ps1 README.txt; do
    cp "${DOCKER_DIR}/${f}" "${OUT_DIR}/"
done

chmod +x "${OUT_DIR}/start-gui.sh" "${OUT_DIR}/stop-gui.sh" "${OUT_DIR}/generate-compose-override.sh"

if command -v python3 >/dev/null 2>&1; then
    python3 "${SCRIPT_DIR}/fix_sh_line_endings.py" "${OUT_DIR}" >/dev/null
fi

if [[ -d "${REPO_ROOT}/config" ]]; then
    echo "[3/4] Copying default config/ ..."
    cp -a "${REPO_ROOT}/config/." "${OUT_DIR}/config/"
fi

echo "[4/4] Done."
echo
echo "Bundle directory: ${OUT_DIR}"
echo "Create zip:       (cd ${REPO_ROOT} && zip -r ${BUNDLE_NAME}.zip ${BUNDLE_NAME})"
echo "User workflow:    unzip → start-gui.bat → http://localhost:8501"
echo
