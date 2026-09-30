#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH --comment="scitas_hwperf"
#SBATCH --chdir /scratch/chouh/breifast_small_test
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/vtune-trio-%j.out

# VTune hotspots, hpc-performance and memory-access on one bench build, over the ITT-resumed region only.
# Args: <exe dir holding bench + libtbb.so.12> <label> [threads] [iterations] [collections] [out dir]
EXE="$1/bench"
LABEL="$2"
THREADS="${3:-72}"
ITERATIONS="${4:-3}"
COLLECTS="${5:-hotspots hpc-performance memory-access}"
MANIFEST_DIR=/scratch/chouh/breifast_small_test/lot_diff36_cpp
REPO=/home/chouh/work/src/breifast-cpp
OUT="${6:-$REPO/profiling/vtune-${LABEL}-threads${THREADS}-job${SLURM_JOB_ID}}"
mkdir -p "$OUT"

module load intel-oneapi-vtune

hostname
echo "exe $EXE ($(stat -c %y "$EXE"))"
{
  echo "host $(hostname), job ${SLURM_JOB_ID}, threads ${THREADS}, iterations ${ITERATIONS}"
  echo "exe $EXE ($(stat -c %y "$EXE"))"
  echo "git $(git -C "$REPO" rev-parse --short HEAD), $(git -C "$REPO" status --porcelain | wc -l) modified file(s)"
} >> "$OUT/provenance.txt"
git -C "$REPO" diff > "$OUT/source.patch"

declare -A KNOBS=(
  [hotspots]="-knob sampling-mode=hw -knob enable-stack-collection=false"
  [hpc-performance]="-knob collect-memory-bandwidth=true -knob enable-stack-collection=false"
  # analyze-mem-objects needs ptrace, which is blocked on these nodes
  [memory-access]=""
)

for COLLECT in $COLLECTS; do
  echo "=== vtune $COLLECT ==="
  # The ITT trace is sometimes written corrupted, which leaves only a sliver of the resumed region: retry.
  for ATTEMPT in 1 2 3; do
    rm -rf "$OUT/$COLLECT"
    vtune -collect "$COLLECT" -start-paused ${KNOBS[$COLLECT]} -result-dir "$OUT/$COLLECT" \
          -- "$EXE" "$MANIFEST_DIR" "$THREADS" "$ITERATIONS" > "$OUT/$COLLECT-collect.log" 2>&1
    grep -E "iteration|vtune: Error" "$OUT/$COLLECT-collect.log"
    grep -q "Data file is corrupted" "$OUT/$COLLECT-collect.log" || break
    echo "attempt $ATTEMPT: corrupted ITT trace, retrying"
  done
  vtune -report summary -result-dir "$OUT/$COLLECT" -report-output "$OUT/$COLLECT-summary.txt" > /dev/null 2>&1
done
