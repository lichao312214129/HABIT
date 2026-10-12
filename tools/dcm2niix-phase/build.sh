#!/bin/bash
# Cross-compile a Windows dcm2niix.exe from the v1.0.20260724 tag.
# The cloud machine clones upstream. This script does not read any
# local medical-image tree.
set -euo pipefail

TAG="v1.0.20260724"
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="${ROOT}/src"

sudo apt-get update
sudo apt-get install -y git cmake make python3 g++-mingw-w64-x86-64 gcc-mingw-w64-x86-64

if [[ ! -d "${SRC}/.git" ]]; then
  git clone --depth 1 --branch "${TAG}" https://github.com/rordenlab/dcm2niix.git "${SRC}"
fi

python3 "${ROOT}/patch_y.py" "${SRC}"

# Pass only cache variables that this tag's CMakeLists actually defines.
cmake_args=()
if grep -q "ZLIB_IMPLEMENTATION" "${SRC}/CMakeLists.txt"; then
  cmake_args+=("-DZLIB_IMPLEMENTATION=Cloudflare")
fi
if grep -q "OpenJPEG_IMPLEMENTATION" "${SRC}/CMakeLists.txt"; then
  cmake_args+=("-DOpenJPEG_IMPLEMENTATION=Internal")
fi
if grep -q "CharLS_IMPLEMENTATION" "${SRC}/CMakeLists.txt"; then
  cmake_args+=("-DCharLS_IMPLEMENTATION=Internal")
fi
if grep -q "option(USE_OPENJPEG" "${SRC}/CMakeLists.txt"; then
  cmake_args+=("-DUSE_OPENJPEG=ON")
fi
if grep -q "option(USE_JPEGLS" "${SRC}/CMakeLists.txt"; then
  cmake_args+=("-DUSE_JPEGLS=ON")
fi

cmake -S "${SRC}" -B "${ROOT}/build-win" \
  -DCMAKE_SYSTEM_NAME=Windows \
  -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
  -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ \
  -DCMAKE_RC_COMPILER=x86_64-w64-mingw32-windres \
  -DCMAKE_EXE_LINKER_FLAGS="-static -static-libgcc -static-libstdc++" \
  "${cmake_args[@]}"

cmake --build "${ROOT}/build-win" --parallel

exe="$(find "${ROOT}/build-win" -type f -name 'dcm2niix.exe' | head -n 1)"
if [[ -z "${exe}" ]]; then
  echo "dcm2niix.exe was not produced" >&2
  exit 1
fi
cp "${exe}" "${ROOT}/dcm2niix.exe"
file "${ROOT}/dcm2niix.exe" || true
ls -l "${ROOT}/dcm2niix.exe"
