#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 0:20:00
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/stream-%j.out

# Achievable DRAM bandwidth reference for the bench numbers: whole node, and one SNC domain.
# stream.c 5.10 from cs.virginia.edu, gcc -O3 -march=icelake-server -fopenmp -DSTREAM_ARRAY_SIZE=100000000 (800 MB/array)
STREAM=/scratch/chouh/stream/stream_omp

hostname
echo "=== whole node, 72 threads, spread ==="
OMP_NUM_THREADS=72 OMP_PROC_BIND=spread OMP_PLACES=cores "$STREAM" | grep -E "Array size|Memory per array|Number of Threads counted|Copy|Scale|Add|Triad"
echo "=== domain 0, 18 threads, local memory ==="
OMP_NUM_THREADS=18 OMP_PROC_BIND=close numactl --cpunodebind=0 --membind=0 "$STREAM" | grep -E "Array size|Number of Threads counted|Copy|Scale|Add|Triad"
