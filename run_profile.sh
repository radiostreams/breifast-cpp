#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH --comment="scitas_hwperf"
#SBATCH --chdir /home/chouh/work/src/breifast-cpp/profiling
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/vtune-pending-%j.out

module load intel-oneapi-vtune
module load hwloc/2.9.1

# Thread count is customizable from the command line: `sbatch run_profile.sh <num_threads>`.
# Defaults to 0, bench's own sentinel for "no tbb::global_control cap" (all available cores).
NUM_THREADS="${1:-0}"
ITERATIONS=5
NPIX=16384
COLLECT=hotspots

# Bakes every parameter that affects the run into the name (plus the job ID) so results and logs
# from different sweeps never collide or need guessing at. Both the VTune result dir and this job's
# own log file (renamed below, since #SBATCH --output can't see $NUM_THREADS at submit time) share
# this same descriptive stem.
STEM="vtune-${COLLECT}-npix${NPIX}-iters${ITERATIONS}-threads${NUM_THREADS}-job${SLURM_JOB_ID}"
RESULT_DIR="${STEM}-$(date +%Y%m%d-%H%M%S)"

# Renaming our own in-progress --output log is safe on Linux: SLURM holds it open by file
# descriptor/inode, not path, so writes keep landing in the same file under its new name.
mv "/home/chouh/work/src/breifast-cpp/profiling/vtune-pending-${SLURM_JOB_ID}.out" \
   "/home/chouh/work/src/breifast-cpp/profiling/${STEM}.out" 2>/dev/null || true

vtune -collect "$COLLECT" -start-paused -knob sampling-mode=hw -knob sampling-interval=100 -result-dir "$RESULT_DIR" -- /home/chouh/work/src/breifast-cpp/builddir/bench /home/chouh/work/src/breifast/small_test/lot_diff36_cpp/ "$NUM_THREADS" "$ITERATIONS"
#vtune -collect threading -start-paused -knob sampling-and-waits=hw -knob sampling-interval=10 -result-dir "$RESULT_DIR" -- /home/chouh/work/src/breifast-cpp/builddir/bench /home/chouh/work/src/breifast/small_test/lot_diff36_cpp/ "$NUM_THREADS" "$ITERATIONS"

# Provenance: pin every result to the exact source state and build flags that produced it, so the
# numbers stay attributable once the tree has moved on. Consumed by notes/analyses/INDEX.md.
# Best-effort throughout -- a missing git or python on the compute node must not fail the run.
REPO=/home/chouh/work/src/breifast-cpp
{
  echo "date         $(date -Is)"
  echo "host         $(hostname)"
  echo "job          ${SLURM_JOB_ID}"
  echo "collect      ${COLLECT} threads=${NUM_THREADS} iters=${ITERATIONS} npix=${NPIX}"
  echo "git-head     $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git-dirty    $(git -C "$REPO" status --porcelain 2>/dev/null | wc -l) file(s) modified"
  echo "bench-built  $(stat -c %y "$REPO/builddir/bench" 2>/dev/null || echo unknown)"
  echo "cxxflags     $(python3 -c "
import json,sys
for e in json.load(open('$REPO/builddir/compile_commands.json')):
    if e['file'].endswith('plane_lot_dispatch.cpp'):
        print(' '.join(t for t in e['command'].split()
                       if t.startswith(('-O','-D','-m','-std=')) or t == '-g')); break
" 2>/dev/null || echo unknown)"
} > "$RESULT_DIR/provenance.txt"

# The full uncommitted diff, so a dirty-tree result is still exactly reconstructible.
git -C "$REPO" diff > "$RESULT_DIR/source.patch" 2>/dev/null || true
