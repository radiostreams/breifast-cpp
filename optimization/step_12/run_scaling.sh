#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH --comment="scitas_hwperf"
#SBATCH --chdir /home/chouh/work/src/breifast-cpp/profiling
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/scaling-pending-%j.out

EXE=/home/chouh/work/src/breifast-cpp/builddir/bench
MANIFEST_DIR=/home/chouh/work/src/breifast/small_test/lot_diff36_cpp/
THREADS="1 2 4 9 18 36 72"
ITEMS_PER_THREAD=15

STEM="scaling-job${SLURM_JOB_ID}"
mv "/home/chouh/work/src/breifast-cpp/profiling/scaling-pending-${SLURM_JOB_ID}.out" \
   "/home/chouh/work/src/breifast-cpp/profiling/${STEM}.out" 2>/dev/null || true

echo "=== strong scaling (fixed 1080 items) ==="
for T in $THREADS; do
  echo "--- threads=$T ---"
  "$EXE" "$MANIFEST_DIR" "$T" 1 1080 2>&1 | grep -E "run_batch_core|iteration 0"
done

echo
echo "=== weak scaling (${ITEMS_PER_THREAD} items/thread) ==="
for T in $THREADS; do
  ITEMS=$((T * ITEMS_PER_THREAD))
  echo "--- threads=$T, items=$ITEMS ---"
  "$EXE" "$MANIFEST_DIR" "$T" 1 "$ITEMS" 2>&1 | grep -E "run_batch_core|iteration 0"
done
