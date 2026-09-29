#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH --comment="scitas_hwperf"
#SBATCH --chdir /home/chouh/work/src/breifast-cpp/profiling
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/numa-isolation-%j.out

module load hwloc/2.9.1

NUMA4=/home/chouh/work/src/breifast-cpp/builddir/bench
SINGLE=/home/chouh/work/src/breifast-cpp/profiling/bench-single-arena/bench
MANIFEST_DIR=/home/chouh/work/src/breifast/small_test/lot_diff36_cpp/
ITEMS_PER_THREAD=15

run() {
  local label=$1 threads=$2; shift 2
  echo "--- $label threads=$threads items=$((threads * ITEMS_PER_THREAD)) ---"
  "$@" "$MANIFEST_DIR" "$threads" 2 $((threads * ITEMS_PER_THREAD)) 2>&1 | grep -E "\[numa\]|iteration|run_batch_core"
}

hostname; lscpu | grep -E "^(Socket|Core|Thread|NUMA node)"; numactl -H | head -2

echo "=== one NUMA domain, cpu+mem bound, single arena ==="
for T in 1 2 4 9 18; do
  run bound "$T" numactl --cpunodebind=0 --membind=0 "$SINGLE"
done

echo "=== whole node, unbound ==="
run single-arena 1 "$SINGLE"
for T in 18 36 72; do
  run single-arena "$T" "$SINGLE"
  run numa4-arenas "$T" "$NUMA4"
done
