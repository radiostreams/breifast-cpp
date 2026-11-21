#!/bin/bash
# Build and install breifast_helpers into a Python venv on a machine without nix (tested on Ubuntu 24.04, gcc 13).
#
#   scripts/install-without-nix.sh <venv> [deps-prefix]
#
# The header-only dependencies the nix flake would supply (xtl, xsimd, xtensor, xtensor-python, nlohmann_json)
# are built from their release tarballs into deps-prefix (default ~/.local/opt/breifast-cpp-deps); meson, ninja,
# pybind11 and a recent cmake come from pip into the venv. TBB is fetched and built by meson itself, as usual.
# The wheel is built with -march=native: install it only on machines of the same CPU family as the build host.
set -eu
VENV=$(realpath "$1")
PREFIX=${2:-$HOME/.local/opt/breifast-cpp-deps}
SRC_DIR=$(cd "$(dirname "$0")/.." && pwd)
PY=$VENV/bin/python
mkdir -p "$PREFIX/src"

$PY -m pip install -q meson meson-python ninja pybind11 cmake
NUMPY_PC=$($PY -c 'import numpy, os; print(os.path.join(os.path.dirname(numpy.__file__), "_core/lib/pkgconfig"))')
PYBIND_PC=$($PY -c 'import pybind11, os; print(os.path.join(os.path.dirname(pybind11.__file__), "share/pkgconfig"))')
PYBIND_CMAKE=$($PY -m pybind11 --cmakedir)

build() { # repo tag dir
  local repo=$1 tag=$2 dir=$3
  if [ ! -f "$PREFIX/src/$dir/.installed" ]; then
    (cd "$PREFIX/src" && rm -rf "$dir" && curl -sL "https://github.com/$repo/archive/refs/tags/$tag.tar.gz" | tar xz)
    "$VENV/bin/cmake" -S "$PREFIX/src/$dir" -B "$PREFIX/src/build-$dir" -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DCMAKE_PREFIX_PATH="$PREFIX;$PYBIND_CMAKE" -DPython_EXECUTABLE="$PY" -DPYTHON_EXECUTABLE="$PY" \
      -DJSON_BuildTests=OFF -DBUILD_TESTS=OFF > "$PREFIX/src/build-$dir.log" 2>&1
    "$VENV/bin/cmake" --build "$PREFIX/src/build-$dir" >> "$PREFIX/src/build-$dir.log" 2>&1
    "$VENV/bin/cmake" --install "$PREFIX/src/build-$dir" >> "$PREFIX/src/build-$dir.log" 2>&1
    touch "$PREFIX/src/$dir/.installed"
  fi
}
build xtensor-stack/xtl 0.8.2 xtl-0.8.2
build xtensor-stack/xsimd 14.3.0 xsimd-14.3.0
build xtensor-stack/xtensor 0.27.2 xtensor-0.27.2
build nlohmann/json v3.12.0 json-3.12.0
build xtensor-stack/xtensor-python 0.29.0 xtensor-python-0.29.0

export PKG_CONFIG_PATH="$PREFIX/share/pkgconfig:$NUMPY_PC:$PYBIND_PC${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export PATH="$VENV/bin:$PATH"
$PY -m pip install --no-build-isolation "$SRC_DIR"
$PY -c "import breifast_helpers as b; print('breifast_helpers installed, spectron kernel v%d' % b.SPECTRON_KERNEL_VERSION)"
