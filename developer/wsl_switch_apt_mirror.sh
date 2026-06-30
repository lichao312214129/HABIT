#!/usr/bin/env bash
# Switch WSL Ubuntu apt to Aliyun mirror (requires sudo password once).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_FILE="${SCRIPT_DIR}/wsl_ubuntu.sources.aliyun"
DEST="/etc/apt/sources.list.d/ubuntu.sources"

if [[ ! -f "${SRC_FILE}" ]]; then
  echo "Missing ${SRC_FILE}" >&2
  exit 1
fi

echo "[apt] Backup current sources..."
sudo mkdir -p /etc/apt/backup
sudo cp "${DEST}" "/etc/apt/backup/ubuntu.sources.bak.$(date +%Y%m%d_%H%M%S)"

echo "[apt] Install Aliyun mirror..."
sudo cp "${SRC_FILE}" "${DEST}"

echo "[apt] Current URIs:"
grep '^URIs:' "${DEST}"

echo "[apt] Updating package index (ForceIPv4)..."
sudo apt-get -o Acquire::ForceIPv4=true update

echo "[apt] Done."
