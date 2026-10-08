#!/bin/bash
#SBATCH --exclusive
#SBATCH -t 2:00:00
#SBATCH --comment="scitas_hwperf"
#SBATCH -p standard,bigmem
#SBATCH --chdir /home/chouh/work/src/breifast-cpp/profiling
#SBATCH --output=/home/chouh/work/src/breifast-cpp/profiling/t72-paired-%j.out

# Paired flat against tiled (tile 2048) band_accumulate, T=72 single arena, 4-channel synthetic lot.
# Rounds alternate the order (flat tile, tile flat, ...) so drift cancels. Args: [rounds] [iterations] [tile]
ROUNDS="${1:-4}"
ITERATIONS="${2:-6}"
TILE="${3:-2048}"
DIR="${DIR:-/home/chouh/work/src/breifast-cpp/profiling/ab-tilesweep}"
MANIFEST_DIR="${MANIFEST_DIR:-/scratch/chouh/synthetic_lot_nf4/}"
ITEMS="${ITEMS:-240}"

run() {
  local label=$1; shift
  "$@" 2>&1 | grep -E "iteration|per-item cost" | sed "s/^/$label /"
}

hostname
for round in $(seq 1 "$ROUNDS"); do
  for variant in $([ $((round % 2)) = 1 ] && echo "flat tile" || echo "tile flat"); do
    if [ "$variant" = flat ]; then
      run "round$round flat" "$DIR/flat/bench" "$MANIFEST_DIR" 72 "$ITERATIONS" "$ITEMS"
    else
      BAND_TILE=$TILE run "round$round tile" "$DIR/tiled/bench" "$MANIFEST_DIR" 72 "$ITERATIONS" "$ITEMS"
    fi
  done
done
