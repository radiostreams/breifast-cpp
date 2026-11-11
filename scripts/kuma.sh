#!/usr/bin/env bash
# Build, test and benchmark this working tree on the kuma cluster, from the laptop.
#
# Usage:
#   scripts/kuma.sh build [release|asserts|all]                        (default: all)
#   scripts/kuma.sh test [release|asserts|all] [-- <meson test args>]  (builds first; default: all)
#   scripts/kuma.sh bench <manifest_dir> [<gpu_bench.py args>]         (builds release first; adds --mode gpu)
#   scripts/kuma.sh shell                                             (interactive dev shell on a GPU node)
#   scripts/kuma.sh sync                                              (only copy the working tree)
#
# Every command first copies the working tree, uncommitted changes included, to a mirror directory on the
# cluster with rsync, then runs scripts/kuma_remote.sh there through srun, inside `nix develop`. Compilation
# runs as its own job step on the build QOS, without a GPU. Tests and the shell then run on the debug QOS and
# benchmarks on the normal QOS, each with one GPU, and never compile (the shell included: run `build` first). All
# steps use the same partition, so -march=native matches the node the code runs on.
#
# The mirror is overwritten to match the laptop, so don't point KUMA_DIR at a clone you edit on the cluster. Build
# directories stay on the cluster between runs: builddir-gpu (release, asserts compiled out) and
# builddir-gpu-asserts (-Db_ndebug=false). Test logs and benchmark JSON are copied back to kuma-logs/<timestamp>/.
#
# Examples:
#   scripts/kuma.sh test                       both build directories, every meson test
#   scripts/kuma.sh test release -- gpu        release build, only the `gpu` test
#   scripts/kuma.sh bench /path/on/kuma/to/lot --plane-count 36
#
# Configuration, from the environment:
#   KUMA_HOST             ssh destination                       (default kuma, the ~/.ssh/config alias)
#   KUMA_DIR              mirror directory on the cluster       (default /work/ska/hchouh/src/breifast-cpp-laptop)
#   KUMA_PARTITION        Slurm partition for both steps        (default h100)
#   KUMA_ACCOUNT          Slurm account, passed only when set
#   KUMA_BUILD_QOS        QOS of the build step                 (default build)
#   KUMA_BUILD_TIME       time limit of the build step          (default 00:30:00)
#   KUMA_BUILD_CPUS       CPU cores of the build step           (default 16)
#   KUMA_BUILD_SRUN_ARGS  extra srun arguments of the build step, word-split
#   KUMA_RUN_QOS          QOS of test and shell                 (default debug)
#   KUMA_RUN_TIME         time limit of test and shell          (default 00:30:00)
#   KUMA_RUN_CPUS         CPU cores of test and shell           (default 8)
#   KUMA_RUN_SRUN_ARGS    extra srun arguments of test and shell, word-split
#   KUMA_BENCH_QOS        QOS of bench                          (default normal)
#   KUMA_BENCH_TIME       time limit of bench                   (default 01:00:00)
#   KUMA_BENCH_CPUS       CPU cores of bench                    (default 16)
#   KUMA_BENCH_SRUN_ARGS  extra srun arguments of bench, word-split
#   KUMA_FIXTURE_DIR      cluster path exported as BREIFAST_FIXTURE_DIR, for the real-data pytest parity test
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

KUMA_HOST="${KUMA_HOST:-kuma}"
KUMA_DIR="${KUMA_DIR:-/work/ska/hchouh/src/breifast-cpp-laptop}"
KUMA_PARTITION="${KUMA_PARTITION:-h100}"
KUMA_BUILD_QOS="${KUMA_BUILD_QOS:-build}"
KUMA_BUILD_TIME="${KUMA_BUILD_TIME:-00:30:00}"
KUMA_BUILD_CPUS="${KUMA_BUILD_CPUS:-16}"
KUMA_RUN_QOS="${KUMA_RUN_QOS:-debug}"
KUMA_RUN_TIME="${KUMA_RUN_TIME:-00:30:00}"
KUMA_RUN_CPUS="${KUMA_RUN_CPUS:-8}"
KUMA_BENCH_QOS="${KUMA_BENCH_QOS:-normal}"
KUMA_BENCH_TIME="${KUMA_BENCH_TIME:-01:00:00}"
KUMA_BENCH_CPUS="${KUMA_BENCH_CPUS:-16}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="kuma-logs/$STAMP"

usage() {
  sed -n '2,/^set -euo pipefail$/{/^set -euo/d;s/^# \{0,1\}//;p;}' "${BASH_SOURCE[0]}"
  exit "${1:-0}"
}

# printf %q of every argument, nothing at all when there is none (a bare `printf '%q '` would emit '').
quote_args() {
  if (($#)); then printf '%q ' "$@"; fi
}

sync_tree() {
  echo ">> syncing working tree to $KUMA_HOST:$KUMA_DIR"
  ssh "$KUMA_HOST" "mkdir -p $(quote_args "$KUMA_DIR")"
  # Excluded paths are also protected from --delete, so the cluster-side build directories and results survive.
  rsync -az --delete \
    --exclude='/builddir*' --exclude='/dist*' --exclude='/kuma-logs/' \
    --exclude='__pycache__/' --exclude='/tests/fixtures/*.npz' \
    ./ "$KUMA_HOST:$KUMA_DIR/"
}

# run_step <build|run|bench> <interactive 0|1> <kuma_remote.sh args...>: runs scripts/kuma_remote.sh as one srun
# step, on the build QOS without a GPU, or on the run or bench QOS with one GPU.
run_step() {
  local kind=$1 interactive=$2
  shift 2
  local srun=(srun "--partition=$KUMA_PARTITION")
  if [[ -n "${KUMA_ACCOUNT:-}" ]]; then srun+=("--account=$KUMA_ACCOUNT"); fi
  # shellcheck disable=SC2206 # the *_SRUN_ARGS variables are meant to be word-split
  case "$kind" in
    build)
      srun+=("--qos=$KUMA_BUILD_QOS" "--cpus-per-task=$KUMA_BUILD_CPUS" "--time=$KUMA_BUILD_TIME")
      srun+=(${KUMA_BUILD_SRUN_ARGS:-})
      ;;
    run)
      srun+=("--qos=$KUMA_RUN_QOS" --gres=gpu:1 "--cpus-per-task=$KUMA_RUN_CPUS" "--time=$KUMA_RUN_TIME")
      srun+=(${KUMA_RUN_SRUN_ARGS:-})
      ;;
    bench)
      srun+=("--qos=$KUMA_BENCH_QOS" --gres=gpu:1 "--cpus-per-task=$KUMA_BENCH_CPUS" "--time=$KUMA_BENCH_TIME")
      srun+=(${KUMA_BENCH_SRUN_ARGS:-})
      ;;
  esac
  local ssh_flags=()
  if [[ "$interactive" == 1 ]]; then
    srun+=(--pty)
    ssh_flags+=(-t)
  fi

  local env_prefix=""
  if [[ -n "${KUMA_FIXTURE_DIR:-}" ]]; then
    env_prefix="BREIFAST_FIXTURE_DIR=$(quote_args "$KUMA_FIXTURE_DIR")"
  fi
  # bash -l: a login shell, so the profile that puts nix on PATH is read on the compute node.
  local command
  command="cd $(quote_args "$KUMA_DIR")&& $env_prefix $(quote_args "${srun[@]}")bash -l scripts/kuma_remote.sh $(quote_args "$@")"
  ssh "${ssh_flags[@]}" "$KUMA_HOST" "$command"
}

fetch_test_logs() {
  mkdir -p "$LOG_DIR"
  if rsync -az --prune-empty-dirs \
    --include='/builddir-gpu*/' --include='/builddir-gpu*/meson-logs/' \
    --include='/builddir-gpu*/meson-logs/testlog.txt' --exclude='*' \
    "$KUMA_HOST:$KUMA_DIR/" "$LOG_DIR/"; then
    echo ">> test logs in $LOG_DIR"
  else
    echo "warning: could not fetch the test logs" >&2
  fi
}

# parse_variant <args...>: sets `variant` (default all) and `rest` (the arguments after it and an optional --).
parse_variant() {
  variant=all
  if [[ $# -gt 0 && "$1" != -- ]]; then
    variant=$1
    shift
  fi
  if [[ "${1:-}" == -- ]]; then shift; fi
  case "$variant" in
    release | asserts | all) ;;
    *) usage 1 ;;
  esac
  rest=("$@")
}

[[ $# -ge 1 ]] || usage 1
command_name=$1
shift

case "$command_name" in
  sync)
    sync_tree
    ;;
  build)
    parse_variant "$@"
    sync_tree
    run_step build 0 build "$variant"
    ;;
  test)
    parse_variant "$@"
    sync_tree
    run_step build 0 build "$variant"
    status=0
    run_step run 0 test "$variant" "${rest[@]}" || status=$?
    fetch_test_logs
    exit "$status"
    ;;
  bench)
    [[ $# -ge 1 ]] || usage 1
    args=("$@")
    has_mode=0
    has_json=0
    for arg in "$@"; do
      case "$arg" in
        --mode | --mode=*) has_mode=1 ;;
        --json | --json=*) has_json=1 ;;
      esac
    done
    if ((has_mode == 0)); then args+=(--mode gpu); fi
    json=""
    if ((has_json == 0)); then
      json="kuma-logs/bench-$STAMP.json"
      args+=(--json "$json")
    fi
    sync_tree
    run_step build 0 build release
    status=0
    run_step bench 0 bench "${args[@]}" || status=$?
    if [[ -n "$json" ]]; then
      mkdir -p "$LOG_DIR"
      if rsync -az "$KUMA_HOST:$KUMA_DIR/$json" "$LOG_DIR/"; then
        echo ">> benchmark results in $LOG_DIR/$(basename "$json")"
      else
        echo "warning: no benchmark results to fetch" >&2
      fi
    fi
    exit "$status"
    ;;
  shell)
    sync_tree
    run_step run 1 shell
    ;;
  -h | --help | help)
    usage
    ;;
  *)
    usage 1
    ;;
esac
