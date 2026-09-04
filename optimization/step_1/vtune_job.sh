#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH --comment="scitas_hwperf"
#SBATCH --output=/work/ska/hchouh/src/breifast-cpp/profiling/vtune-pending-%j.out
#
# VTune hotspots collection for one already-built `bench`. Submitted by profile_rev.sh, which
# passes the binary and a label identifying the revision it was built from.
#
#   sbatch vtune_job.sh <bench-exe> <label> [num_threads] [iterations]

set -euo pipefail

EXE="$1"
LABEL="$2"
NUM_THREADS="${3:-72}"
ITERATIONS="${4:-5}"

REPO=/work/ska/hchouh/src/breifast-cpp
DATASET=/home/chouh/work/src/breifast/small_test/lot_diff36_cpp/
PROFILING="$REPO/profiling"
NPIX=16384
COLLECT=hotspots

module load intel-oneapi-vtune

# Same naming convention as run_profile.sh, with the revision label added so results from
# different points in the optimization history never collide.
STEM="vtune-${COLLECT}-${LABEL}-npix${NPIX}-iters${ITERATIONS}-threads${NUM_THREADS}-job${SLURM_JOB_ID}"
RESULT_DIR="$PROFILING/${STEM}-$(date +%Y%m%d-%H%M%S)"

# SLURM holds its --output file open by inode, so renaming it mid-run keeps the log flowing
# into the same file under a name that says what the run was.
mv "$PROFILING/vtune-pending-${SLURM_JOB_ID}.out" "$PROFILING/${STEM}.out" 2>/dev/null || true

# -start-paused waits for the target to call __itt_resume() before collecting anything.
# Revisions before the ITT instrumentation was added (e.g. perf-step_1-before / 1f30b6e)
# never call it, so the collector stays paused for the entire run and the result is an
# empty ~6 MB capture with zero samples -- happened twice (jobs 66343033, 66347252) before
# this check was added. Detect ITT support in the actual binary being profiled instead of
# assuming it.
PAUSE_FLAG=()
if nm "$EXE" 2>/dev/null | grep -q '__itt_resume_ptr'; then
  PAUSE_FLAG=(-start-paused)
else
  echo "warning: $EXE has no ITT instrumentation -- profiling the whole process, not just the parallel region" >&2
fi

vtune -collect "$COLLECT" "${PAUSE_FLAG[@]}" \
      -knob sampling-mode=hw -knob sampling-interval=100 \
      -result-dir "$RESULT_DIR" \
      -- "$EXE" "$DATASET" "$NUM_THREADS" "$ITERATIONS"

# Provenance travels with the result, so the numbers stay attributable once the tree moves on.
{
  echo "date         $(date -Is)"
  echo "host         $(hostname)"
  echo "job          ${SLURM_JOB_ID}"
  echo "label        ${LABEL}"
  echo "collect      ${COLLECT} threads=${NUM_THREADS} iters=${ITERATIONS} npix=${NPIX}"
  echo "exe          ${EXE}"
  echo "exe-built    $(stat -c %y "$EXE" 2>/dev/null || echo unknown)"
  echo "dataset      ${DATASET}"
  cat "$(dirname "$EXE")/../.step_provenance" 2>/dev/null || true
} > "$RESULT_DIR/provenance.txt"
