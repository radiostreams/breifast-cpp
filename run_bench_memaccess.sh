#!/bin/bash
#SBATCH --exclusive
#SBATCH --comment="scitas_hwperf"
#SBATCH -t 1:00:00
#SBATCH -p standard,bigmem
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/breifast-bench-memaccess-%j.out

MANIFEST_DIR=/home/chouh/work/src/breifast/small_test/lot_diff36_cpp/
EXE_PATH=/home/chouh/work/src/breifast-cpp/builddir/bench
RESULT_DIR=/home/chouh/work/src/breifast-cpp/profiling/breifast_bench_memaccess_${SLURM_JOB_ID}

module load intel-oneapi-vtune
module load hwloc/2.9.1

vtune -collect memory-access             \
      -result-dir "$RESULT_DIR"          \
      -finalization-mode=full            \
      -- $EXE_PATH $MANIFEST_DIR 72 10
