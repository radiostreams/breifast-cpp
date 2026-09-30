#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH --comment="scitas_hwperf"
#SBATCH --chdir /home/chouh/work/src/breifast-cpp/profiling
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/ab-%j.out

# Same-node A/B: binaries in profiling/ab/before and profiling/ab/after, interleaved A B B A A B.
AB_DIR="${2:-/home/chouh/work/src/breifast-cpp/profiling/ab}"
MANIFEST_DIR=/home/chouh/work/src/breifast/small_test/lot_diff36_cpp/
ITERATIONS="${1:-5}"

run() {
  local label=$1 config=$2; shift 2
  "$@" 2>&1 | grep -E "iteration|per-item cost" | sed "s/^/$label $config /"
}

hostname
for LABEL in before after after before before after; do
  EXE="$AB_DIR/$LABEL/bench"
  run "$LABEL" T18-bound numactl --cpunodebind=0 --membind=0 "$EXE" "$MANIFEST_DIR" 18 "$ITERATIONS" 270
  run "$LABEL" T72-single "$EXE" "$MANIFEST_DIR" 72 "$ITERATIONS" 1080
done
