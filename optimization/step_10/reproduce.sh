#!/bin/bash
#SBATCH --exclusive
#SBATCH --comment="scitas_hwperf"
#SBATCH -t 1:00:00
#SBATCH -p standard,bigmem
#SBATCH --output=/home/chouh/work/src/breifast-cpp/optimization/step_10/reproduce-%j.out

# Reproduces the step 10 zone profile: a Tracy capture of bench at 72 threads, exported to the
# two CSVs checked in beside this script.
#
# Prerequisites, both outside the repo's build:
#   1. Tracy 0.14.2 (81c50055) checked out at <repo>/tracy, with capture and csvexport built:
#        git clone https://github.com/wolfpld/tracy tracy && git -C tracy checkout 81c50055
#        cmake -B tracy/capture/build   -S tracy/capture   -DCMAKE_BUILD_TYPE=Release && cmake --build tracy/capture/build -j
#        cmake -B tracy/csvexport/build -S tracy/csvexport -DCMAKE_BUILD_TYPE=Release && cmake --build tracy/csvexport/build -j
#   2. An instrumented bench, built in the dev shell:
#        meson setup builddir-tracy -Dtracy=true
#        ninja -C builddir-tracy bench
#
# Then:  sbatch optimization/step_10/reproduce.sh
set -uo pipefail

NUM_THREADS="${1:-72}"
ITERATIONS="${2:-3}"

REPO=/home/chouh/work/src/breifast-cpp
STEP="$REPO/optimization/step_10"
MANIFEST_DIR=/home/chouh/work/src/breifast/small_test/lot_diff36_cpp/
EXE="$REPO/builddir-tracy/bench"
CAPTURE="$REPO/tracy/capture/build/tracy-capture"
CSVEXPORT="$REPO/tracy/csvexport/build/tracy-csvexport"
TRACE="$STEP/bench-threads${NUM_THREADS}-iters${ITERATIONS}-job${SLURM_JOB_ID}.tracy"

for tool in "$EXE" "$CAPTURE" "$CSVEXPORT"; do
  if [ ! -x "$tool" ]; then
    echo "missing: $tool (see the prerequisites at the top of this script)" >&2
    exit 1
  fi
done

echo "node=$(hostname) threads=$NUM_THREADS iterations=$ITERATIONS"
echo "git-head $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" status --porcelain | wc -l) file(s) modified)"
echo

# Capture first: it retries until the client appears, so this ordering cannot miss the start.
# bench is built with TRACY_NO_EXIT, so it blocks at exit until the trace has drained. Running
# the binary without a tracy-capture alongside it will hang.
"$CAPTURE" -o "$TRACE" -f &
CAPTURE_PID=$!

"$EXE" "$MANIFEST_DIR" "$NUM_THREADS" "$ITERATIONS"

wait $CAPTURE_PID

"$CSVEXPORT" -e "$TRACE" > "$STEP/zones-self-time.csv"
"$CSVEXPORT"    "$TRACE" > "$STEP/zones-total-time.csv"

echo
echo "trace   $TRACE"
echo "self    $STEP/zones-self-time.csv"
echo "total   $STEP/zones-total-time.csv"
echo
echo "=== self time per zone ==="
python3 - "$STEP/zones-self-time.csv" << 'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['name']]
total = sum(int(r['total_ns']) for r in rows)
rows.sort(key=lambda r: -int(r['total_ns']))
print(f'{"zone":<34}{"% self":>9}{"calls":>9}  src')
for r in rows:
    ns = int(r['total_ns'])
    src = r['src_file'].split('/')[-1] + ':' + r['src_line']
    print(f"{r['name']:<34}{100*ns/total:>8.2f}%{int(r['counts']):>9}  {src}")
PY
