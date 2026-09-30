#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH --chdir /scratch/chouh/breifast_small_test
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/python-vs-cpp-%j.out

# Same node, same 1080-plane lot: bench (single arena) vs breifast.py's process_plane on a thread pool.
THREADS="${1:-72}"
EXE=/home/chouh/work/src/breifast-cpp/profiling/ab-noblock/after/bench
PY=/work/ska/hchouh/src/breifast/.venv/bin/python
export OMP_NUM_THREADS=1

hostname
"$EXE" lot_diff36_cpp "$THREADS" 5 2>&1 | grep -E "iteration|mean"
"$PY" /home/chouh/work/src/breifast-cpp/scripts/python_bench.py lot_diff36.pkl "$THREADS" 3
"$EXE" lot_diff36_cpp "$THREADS" 5 2>&1 | grep -E "iteration|mean"
