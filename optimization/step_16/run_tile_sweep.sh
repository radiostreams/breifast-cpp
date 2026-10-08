#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH --comment="scitas_hwperf"
#SBATCH -p standard,bigmem
#SBATCH --chdir /home/chouh/work/src/breifast-cpp/profiling
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/tile-sweep-%j.out

# Tile-size sweep of the tiled band_accumulate on the 4-channel synthetic lot, T=72 single arena.
# The tiled bench reads BAND_TILE (floats); "flat" is the flat-loop build. Two rounds, the second in reverse order.
# Args: [iterations] [tile list]
ITERATIONS="${1:-3}"
TILES="${2:-64 256 1024 2048 4096 8192 16384 32768}"
DIR=/home/chouh/work/src/breifast-cpp/profiling/ab-tilesweep
MANIFEST_DIR=/scratch/chouh/synthetic_lot_nf4/

run() {
  local label=$1; shift
  "$@" 2>&1 | grep -E "iteration|per-item cost" | sed "s/^/$label /"
}

hostname
ORDER="flat $TILES"
REVERSED=$(echo $ORDER | tr ' ' '\n' | tac | tr '\n' ' ')
for round in 1 2; do
  for tile in $([ $round = 1 ] && echo $ORDER || echo $REVERSED); do
    if [ "$tile" = flat ]; then
      run "round$round flat" "$DIR/flat/bench" "$MANIFEST_DIR" 72 "$ITERATIONS" 240
    else
      BAND_TILE=$tile run "round$round tile=$tile" "$DIR/tiled/bench" "$MANIFEST_DIR" 72 "$ITERATIONS" 240
    fi
  done
done
