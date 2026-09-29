#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH --comment="scitas_hwperf"
#SBATCH --chdir /home/chouh/work/src/breifast-cpp/profiling
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/counters-%j.out

module load hwloc/2.9.1

NUMA4=/home/chouh/work/src/breifast-cpp/builddir/bench
SINGLE=/home/chouh/work/src/breifast-cpp/profiling/bench-single-arena-v2/bench
MANIFEST_DIR=/home/chouh/work/src/breifast/small_test/lot_diff36_cpp/
ITEMS_PER_THREAD=15

run() {
  local label=$1 threads=$2; shift 2
  echo "--- $label threads=$threads items=$((threads * ITEMS_PER_THREAD)) ---"
  "$@" "$MANIFEST_DIR" "$threads" 2 $((threads * ITEMS_PER_THREAD)) 2>&1 | grep -E "\[numa\]|iteration|cpu:|vmstat:"
}

hostname
run bound-local 1 numactl --cpunodebind=0 --membind=0 "$SINGLE"
run bound-local 18 numactl --cpunodebind=0 --membind=0 "$SINGLE"
run bound-interleaved 18 numactl --cpunodebind=0 --interleave=all "$SINGLE"
run unbound-single-arena 18 "$SINGLE"
run unbound-single-arena 72 "$SINGLE"
run unbound-numa4-arenas 72 "$NUMA4"
