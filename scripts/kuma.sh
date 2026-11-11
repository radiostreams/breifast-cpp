#!/usr/bin/env bash
# Build, test and benchmark this working tree on the kuma cluster, from the laptop.
#
# Usage:
#   scripts/kuma.sh build [release|asserts|all]                        (default: all)
#   scripts/kuma.sh test [release|asserts|all] [-- <meson test args>]  (builds first; default: all)
#   scripts/kuma.sh bench <manifest_dir> [<gpu_bench.py args>]         (builds release first; adds --mode gpu)
#   scripts/kuma.sh logs [<run>]                                      (job states, then fetch the run's logs)
#   scripts/kuma.sh wait [<run>]                                      (wait for the run to end, then fetch)
#   scripts/kuma.sh shell                                             (interactive dev shell on a GPU node)
#   scripts/kuma.sh sync                                              (only copy the working tree)
#
# build, test and bench copy the working tree, uncommitted changes included, to a mirror directory on the cluster with
# rsync, queue batch jobs with sbatch and return at once: nothing depends on the laptop staying awake. Compilation is
# its own job on the build QOS, without a GPU. Tests then run on the debug QOS and benchmarks on the normal QOS, each
# with one GPU, only if the build succeeded, and never compile. All jobs use the same partition, so -march=native
# matches the node the code runs on. Each submission is a run, named by its timestamp: its job ids, job outputs, test
# logs and benchmark JSON land in kuma-runs/<run>/ on the cluster, and `logs` or `wait` copy them to kuma-logs/<run>/.
# <run> defaults to the latest one.
#
# Every run builds in the same mirror, so a submission refuses while an earlier run still has jobs queued or running:
# syncing would change the tree under them. KUMA_FORCE=1 submits anyway.
#
# The mirror is overwritten to match the laptop, so don't point KUMA_DIR at a clone you edit on the cluster. Build
# directories stay on the cluster between runs: builddir-gpu (release, asserts compiled out) and builddir-gpu-asserts
# (-Db_ndebug=false).
#
# Examples:
#   scripts/kuma.sh test                       both build directories, every meson test
#   scripts/kuma.sh test release -- gpu        release build, only the `gpu` test
#   scripts/kuma.sh wait                       block until the latest run ends, then fetch its logs
#   scripts/kuma.sh bench /path/on/kuma/to/lot --plane-count 36
#
# Configuration, from the environment:
#   KUMA_HOST              ssh destination                      (default kuma, the ~/.ssh/config alias)
#   KUMA_DIR               mirror directory on the cluster      (default /work/ska/hchouh/src/breifast-cpp-laptop)
#   KUMA_PARTITION         Slurm partition of every job         (default h100)
#   KUMA_ACCOUNT           Slurm account, passed only when set
#   KUMA_BUILD_QOS         QOS of the build job                 (default build)
#   KUMA_BUILD_TIME        time limit of the build job          (default 00:30:00)
#   KUMA_BUILD_CPUS        CPU cores of the build job           (default 16)
#   KUMA_BUILD_SLURM_ARGS  extra sbatch arguments of the build job, word-split
#   KUMA_RUN_QOS           QOS of test and shell                (default debug)
#   KUMA_RUN_TIME          time limit of test and shell         (default 00:30:00)
#   KUMA_RUN_CPUS          CPU cores of test and shell          (default 8)
#   KUMA_RUN_SLURM_ARGS    extra sbatch/srun arguments of test and shell, word-split
#   KUMA_BENCH_QOS         QOS of bench                         (default normal)
#   KUMA_BENCH_TIME        time limit of bench                  (default 01:00:00)
#   KUMA_BENCH_CPUS        CPU cores of bench                   (default 16)
#   KUMA_BENCH_SLURM_ARGS  extra sbatch arguments of bench, word-split
#   KUMA_FIXTURE_DIR       cluster path exported as BREIFAST_FIXTURE_DIR, for the real-data pytest parity test
#   KUMA_FORCE             1 to submit while an earlier run still has jobs
#   KUMA_POLL_SECONDS      interval of `wait`                   (default 30)
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
KUMA_POLL_SECONDS="${KUMA_POLL_SECONDS:-30}"

# Job names of every batch job this script queues, so that the guard and `wait` can find them.
JOB_PREFIX=breifast-kuma
RUNS=kuma-runs

usage() {
  sed -n '2,/^set -euo pipefail$/{/^set -euo/d;s/^# \{0,1\}//;p;}' "${BASH_SOURCE[0]}"
  exit "${1:-0}"
}

# printf %q of every argument, nothing at all when there is none (a bare `printf '%q '` would emit '').
quote_args() {
  if (($#)); then printf '%q ' "$@"; fi
}

remote() {
  ssh "$KUMA_HOST" "cd $(quote_args "$KUMA_DIR")&& $1"
}

sync_tree() {
  echo ">> syncing working tree to $KUMA_HOST:$KUMA_DIR"
  ssh "$KUMA_HOST" "mkdir -p $(quote_args "$KUMA_DIR")"
  # Excluded paths are also protected from --delete, so the cluster-side build directories and runs survive.
  rsync -az --delete \
    --exclude='/builddir*' --exclude='/dist*' --exclude='/kuma-logs/' --exclude="/$RUNS/" \
    --exclude='__pycache__/' --exclude='/tests/fixtures/*.npz' \
    ./ "$KUMA_HOST:$KUMA_DIR/"
}

# Refuses to submit while a job of an earlier run is queued or running, unless KUMA_FORCE=1.
require_idle() {
  if [[ "${KUMA_FORCE:-0}" == 1 ]]; then return; fi
  local active
  active="$(ssh "$KUMA_HOST" "squeue --me --noheader --format='%i %j %T' | grep ' $JOB_PREFIX-' || true")"
  if [[ -n "$active" ]]; then
    echo "kuma.sh: an earlier run still has jobs; syncing now would change the tree under them:" >&2
    echo "$active" >&2
    echo "wait for them (scripts/kuma.sh wait), cancel them, or set KUMA_FORCE=1" >&2
    exit 1
  fi
}

# slurm_args <build|run|bench>: the partition, QOS, resources and extra arguments of a job kind.
slurm_args() {
  local args=("--partition=$KUMA_PARTITION")
  if [[ -n "${KUMA_ACCOUNT:-}" ]]; then args+=("--account=$KUMA_ACCOUNT"); fi
  # shellcheck disable=SC2206 # the *_SLURM_ARGS variables are meant to be word-split
  case "$1" in
    build) args+=("--qos=$KUMA_BUILD_QOS" "--cpus-per-task=$KUMA_BUILD_CPUS" "--time=$KUMA_BUILD_TIME" ${KUMA_BUILD_SLURM_ARGS:-}) ;;
    run) args+=("--qos=$KUMA_RUN_QOS" --gres=gpu:1 "--cpus-per-task=$KUMA_RUN_CPUS" "--time=$KUMA_RUN_TIME" ${KUMA_RUN_SLURM_ARGS:-}) ;;
    bench) args+=("--qos=$KUMA_BENCH_QOS" --gres=gpu:1 "--cpus-per-task=$KUMA_BENCH_CPUS" "--time=$KUMA_BENCH_TIME" ${KUMA_BENCH_SLURM_ARGS:-}) ;;
  esac
  quote_args "${args[@]}"
}

# submit <run> <build|run|bench> <name> <dependency job id or ""> <kuma_remote.sh args...>: queues
# scripts/kuma_remote.sh as a batch job of the run, logging to kuma-runs/<run>/<name>.out, and prints its job id.
submit() {
  local run=$1 kind=$2 name=$3 after=$4
  shift 4
  local env="KUMA_RUN_DIR=$(quote_args "$RUNS/$run")"
  if [[ -n "${KUMA_FIXTURE_DIR:-}" ]]; then env+=" BREIFAST_FIXTURE_DIR=$(quote_args "$KUMA_FIXTURE_DIR")"; fi
  # bash -l: a login shell, so the profile that puts nix on PATH is read on the compute node.
  local wrap="$env bash -l scripts/kuma_remote.sh $(quote_args "$@")"
  local sbatch="sbatch --parsable --job-name=$JOB_PREFIX-$name --chdir=$(quote_args "$KUMA_DIR")"
  sbatch+=" --output=$(quote_args "$RUNS/$run/$name.out") $(slurm_args "$kind")"
  if [[ -n "$after" ]]; then sbatch+=" --dependency=afterok:$after --kill-on-invalid-dep=yes"; fi
  remote "$sbatch --wrap=$(quote_args "$wrap")" | cut -d';' -f1
}

# start_run: prints a new run name and creates its directory on the cluster.
start_run() {
  local run
  run="$(date +%Y%m%d-%H%M%S)"
  remote "mkdir -p $(quote_args "$RUNS/$run")" >/dev/null
  echo "$run"
}

# record_jobs <run> <job ids...>: keeps the run's job ids, for `logs` and `wait`.
record_jobs() {
  local run=$1
  shift
  remote "echo $(quote_args "$@")> $(quote_args "$RUNS/$run/jobs")"
  echo ">> run $run: jobs $*"
  echo ">> scripts/kuma.sh wait   (or logs) to fetch kuma-logs/$run when it ends"
}

# resolve_run [<run>]: the given run, or the latest one on the cluster.
resolve_run() {
  if [[ -n "${1:-}" ]]; then
    echo "$1"
  else
    remote "ls -1 $RUNS 2>/dev/null | tail -n 1"
  fi
}

# job_states <run>: one "id name state elapsed" line per job of the run.
job_states() {
  remote "sacct --noheader --parsable2 --allocations --format=JobID,JobName,State,Elapsed \
--jobs=\$(tr ' ' ',' < $(quote_args "$RUNS/$1/jobs"))" | tr '|' ' '
}

fetch_run() {
  local run=$1
  mkdir -p "kuma-logs/$run"
  rsync -az "$KUMA_HOST:$KUMA_DIR/$RUNS/$run/" "kuma-logs/$run/"
  echo ">> logs in kuma-logs/$run"
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
    require_idle
    sync_tree
    run="$(start_run)"
    build_id="$(submit "$run" build build "" build "$variant")"
    record_jobs "$run" "$build_id"
    ;;
  test)
    parse_variant "$@"
    require_idle
    sync_tree
    run="$(start_run)"
    build_id="$(submit "$run" build build "" build "$variant")"
    test_id="$(submit "$run" run test "$build_id" test "$variant" "${rest[@]}")"
    record_jobs "$run" "$build_id" "$test_id"
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
    require_idle
    sync_tree
    run="$(start_run)"
    if ((has_json == 0)); then args+=(--json "$RUNS/$run/bench.json"); fi
    build_id="$(submit "$run" build build "" build release)"
    bench_id="$(submit "$run" bench bench "$build_id" bench "${args[@]}")"
    record_jobs "$run" "$build_id" "$bench_id"
    ;;
  logs)
    run="$(resolve_run "${1:-}")"
    [[ -n "$run" ]] || { echo "kuma.sh: no run on the cluster yet" >&2; exit 1; }
    echo ">> run $run"
    job_states "$run"
    fetch_run "$run"
    ;;
  wait)
    run="$(resolve_run "${1:-}")"
    [[ -n "$run" ]] || { echo "kuma.sh: no run on the cluster yet" >&2; exit 1; }
    echo ">> waiting for run $run"
    while job_states "$run" | grep -qE ' (PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED) '; do
      sleep "$KUMA_POLL_SECONDS"
    done
    states="$(job_states "$run")"
    echo "$states"
    fetch_run "$run"
    # Succeeds only if every job of the run completed.
    ! grep -vq ' COMPLETED ' <<<"$states"
    ;;
  shell)
    sync_tree
    # Interactive, so still srun --pty over a terminal.
    # shellcheck disable=SC2046 # slurm_args prints already-quoted words
    ssh -t "$KUMA_HOST" "cd $(quote_args "$KUMA_DIR")&& srun $(slurm_args run) --pty bash -l scripts/kuma_remote.sh shell"
    ;;
  -h | --help | help)
    usage
    ;;
  *)
    usage 1
    ;;
esac
