#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH --comment="scitas_hwperf"
#SBATCH --output=/work/ska/hchouh/src/breifast-cpp/profiling/scaling-pending-%j.out
#
# Thread-scaling sweep for one already-built `bench`. Submitted by scaling_rev.sh (worktree
# revisions) or directly against the main checkout's builddir, which pass the binary and a
# label identifying the revision it was built from.
#
#   sbatch scaling_job.sh <bench-exe> <label> <dataset-dir> <iterations> "<threads, space-separated>"

set -euo pipefail

EXE="$1"
LABEL="$2"
DATASET="$3"
ITERATIONS="$4"
read -ra THREADS <<< "$5"

PROFILING=/work/ska/hchouh/src/breifast-cpp/profiling
STEM="scaling-${LABEL}-job${SLURM_JOB_ID}"
mv "$PROFILING/scaling-pending-${SLURM_JOB_ID}.out" "$PROFILING/${STEM}.out" 2>/dev/null || true

CSV="$PROFILING/${STEM}.csv"
echo "label,threads,iterations,mean_seconds,total_seconds" > "$CSV"

for T in "${THREADS[@]}"; do
  echo "=== threads=${T} ==="
  OUT=$("$EXE" "$DATASET" "$T" "$ITERATIONS")
  echo "$OUT"
  SUMMARY=$(echo "$OUT" | grep -E "^run_batch_core:")
  TOTAL=$(echo "$SUMMARY" | sed -E 's/.*total = ([0-9.]+)s.*/\1/')
  MEAN=$(echo "$SUMMARY" | sed -E 's/.*mean = ([0-9.]+)s.*/\1/')
  echo "${LABEL},${T},${ITERATIONS},${MEAN},${TOTAL}" >> "$CSV"
done

# Provenance travels with the result, so the numbers stay attributable once the tree moves on.
{
  echo "date        $(date -Is)"
  echo "host        $(hostname)"
  echo "job         ${SLURM_JOB_ID}"
  echo "label       ${LABEL}"
  echo "exe         ${EXE}"
  echo "exe-built   $(stat -c %y "$EXE" 2>/dev/null || echo unknown)"
  echo "dataset     ${DATASET}"
  echo "threads     ${THREADS[*]}"
  echo "iterations  ${ITERATIONS}"
  cat "$(dirname "$EXE")/../.step_provenance" 2>/dev/null || true
} > "$PROFILING/${STEM}.provenance.txt"

echo "csv written to $CSV"
