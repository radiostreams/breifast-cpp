#!/usr/bin/env bash
# Runs on a kuma compute node, started by scripts/kuma.sh through srun; not meant to be called by hand.
#
#   kuma_remote.sh build <release|asserts|all>                     (build QOS: configures and compiles)
#   kuma_remote.sh test <release|asserts|all> [<meson test args>]  (debug QOS: runs tests, never compiles)
#   kuma_remote.sh bench <gpu_bench.py args>                       (normal QOS: needs a built builddir-gpu)
#   kuma_remote.sh shell                                           (debug QOS)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Re-run inside the flake's dev shell, which provides nvcc, meson and the Python environment.
if [[ -z "${IN_NIX_SHELL:-}" ]]; then
  exec nix develop --command bash "${BASH_SOURCE[0]}" "$@"
fi

gpu_info="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'no GPU visible')"
echo ">> $(hostname): $gpu_info"

# Build directories of a variant, with their extra `meson setup` arguments after a colon.
build_dirs() {
  case "$1" in
    release) echo "builddir-gpu:" ;;
    asserts) echo "builddir-gpu-asserts:-Db_ndebug=false" ;;
    all) printf '%s\n' "builddir-gpu:" "builddir-gpu-asserts:-Db_ndebug=false" ;;
    *)
      echo "kuma_remote.sh: unknown variant '$1'" >&2
      return 2
      ;;
  esac
}

require_built() {
  if [[ ! -f "$1/build.ninja" ]]; then
    echo "kuma_remote.sh: $1 is not built; run scripts/kuma.sh build first" >&2
    exit 1
  fi
}

mode=$1
shift

case "$mode" in
  build)
    entries="$(build_dirs "$1")"
    status=0
    while IFS=: read -r dir setup_args; do
      echo ">> building $dir ${setup_args:+($setup_args)}"
      if [[ ! -f "$dir/build.ninja" ]]; then
        # shellcheck disable=SC2086 # setup_args is empty or a single option
        meson setup "$dir" -Dgpu=enabled $setup_args </dev/null || { status=1; continue; }
      fi
      meson compile -C "$dir" </dev/null || status=1
    done <<<"$entries"
    exit "$status"
    ;;
  test)
    entries="$(build_dirs "$1")"
    shift
    status=0
    while IFS=: read -r dir _; do
      require_built "$dir"
      echo ">> testing $dir"
      meson test -C "$dir" --no-rebuild --print-errorlogs "$@" </dev/null || status=1
    done <<<"$entries"
    exit "$status"
    ;;
  bench)
    require_built builddir-gpu
    mkdir -p kuma-logs
    PYTHONPATH="$PWD/builddir-gpu${PYTHONPATH:+:$PYTHONPATH}" python3 scripts/gpu_bench.py "$@"
    ;;
  shell)
    echo ">> dev shell in $PWD on the debug QOS: don't compile here, use scripts/kuma.sh build"
    exec bash -i
    ;;
  *)
    echo "kuma_remote.sh: unknown mode '$mode'" >&2
    exit 2
    ;;
esac
