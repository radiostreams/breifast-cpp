#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH --comment="scitas_hwperf"
#SBATCH --chdir /scratch/chouh/breifast_small_test
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/assess-%j.out

# CPU assessment of one bench build (single arena): strong/weak scaling, then VTune hpc-performance and
# uarch-exploration at 72 threads over the ITT-resumed region only.
# Args: <exe dir holding bench + libtbb.so.12> <label> [collections]. SKIP_SCALING=1 runs VTune only.
EXE="$1/bench"
LABEL="$2"
COLLECTS="${3:-hpc-performance uarch-exploration}"
MANIFEST_DIR=/scratch/chouh/breifast_small_test/lot_diff36_cpp
OUT=/home/chouh/work/src/breifast-cpp/profiling/assess-${LABEL}-${SLURM_JOB_ID}
THREADS="1 2 4 9 18 36 72"
mkdir -p "$OUT"

module load intel-oneapi-vtune

run() {
  local tag=$1 threads=$2 items=$3
  echo "--- $tag threads=$threads items=$items ---"
  "$EXE" "$MANIFEST_DIR" "$threads" 2 "$items" 2>&1 | grep -E "iteration|cpu:|vmstat:"
}

hostname
echo "exe $EXE ($(stat -c %y "$EXE"))"
if [ -z "$SKIP_SCALING" ]; then
  echo "=== strong scaling (1080 items) ==="
  for T in $THREADS; do run strong "$T" 1080; done
  echo "=== weak scaling (15 items/thread) ==="
  for T in $THREADS; do run weak "$T" $((T * 15)); done
fi

for COLLECT in $COLLECTS; do
  echo "=== vtune $COLLECT, 72 threads, 3 iterations ==="
  KNOBS=(-knob enable-stack-collection=false)
  [ "$COLLECT" = hpc-performance ] && KNOBS+=(-knob collect-memory-bandwidth=true)
  # The ITT trace is sometimes written corrupted, which leaves only a sliver of the resumed region: retry.
  for ATTEMPT in 1 2 3; do
    rm -rf "$OUT/$COLLECT"
    vtune -collect "$COLLECT" -start-paused "${KNOBS[@]}" -result-dir "$OUT/$COLLECT" \
          -- "$EXE" "$MANIFEST_DIR" 72 3 > "$OUT/$COLLECT-collect.log" 2>&1
    grep -E "iteration|vtune: Error" "$OUT/$COLLECT-collect.log"
    grep -q "Data file is corrupted" "$OUT/$COLLECT-collect.log" || break
    echo "attempt $ATTEMPT: corrupted ITT trace, retrying"
  done
  vtune -report summary -result-dir "$OUT/$COLLECT" -report-output "$OUT/$COLLECT-summary.txt"
  vtune -report hotspots -group-by function -result-dir "$OUT/$COLLECT" -format csv -csv-delimiter comma \
        -report-output "$OUT/$COLLECT-functions.csv"
done
