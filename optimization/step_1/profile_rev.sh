#!/bin/bash
#
# Reproducibly profile one revision of this repo.
#
#   ./profile_rev.sh <git-rev> [num_threads] [iterations]
#   ./profile_rev.sh perf-step_1-before        # the pre-migration baseline
#
# Checks the revision out into a throwaway worktree, applies build-fix.patch, builds `bench`
# with nix, and submits vtune_job.sh against that binary.
#
# Why the patch: NO commit in this repository configures on its own. meson.build references
# itt_dep, itt_root and tbbmalloc_dep at lines 82/99/125 but none of them has ever been
# defined in any commit (`git log -S 'itt_dep ='` finds nothing) -- the build has only ever
# worked from a dirty working tree. build-fix.patch carries the minimum needed to configure,
# and deliberately does NOT touch buildtype, so a revision is profiled with the flags it was
# actually committed with.

set -euo pipefail

REPO=/work/ska/hchouh/src/breifast-cpp
STEP_DIR="$REPO/optimization/step_1"
WORKTREES=/work/ska/hchouh/src/breifast-cpp-worktrees

REV="${1:-perf-step_1-before}"
NUM_THREADS="${2:-72}"
ITERATIONS="${3:-5}"

SAFE_REV="${REV//\//_}"
WT="$WORKTREES/$SAFE_REV"

# A stale worktree would silently profile the wrong source, so always rebuild it from scratch.
if [ -d "$WT" ]; then
  echo "removing existing worktree $WT"
  git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
fi
mkdir -p "$WORKTREES"
git -C "$REPO" worktree add --detach "$WT" "$REV"

RESOLVED=$(git -C "$WT" rev-parse HEAD)
echo "worktree $WT at $RESOLVED ($REV)"

BUILD_PATCH_APPLIED=false
if grep -q "itt_dep = declare_dependency" "$WT/meson.build"; then
  echo "meson.build already defines itt_dep (commit a303c96 or later) -- skipping build-fix.patch"
else
  git -C "$WT" apply "$STEP_DIR/build-fix.patch"
  BUILD_PATCH_APPLIED=true
  echo "applied build-fix.patch"
fi

# The dev shell comes from the main checkout's flake: the worktree is now dirty from the patch,
# and this keeps one definition of the toolchain across every revision profiled.
nix develop "$REPO" --command meson setup "$WT/builddir" "$WT"
nix develop "$REPO" --command ninja -C "$WT/builddir" bench

# Read back by vtune_job.sh and copied into the result's provenance.txt.
cat > "$WT/.step_provenance" <<PROV
rev-requested  ${REV}
rev-resolved   ${RESOLVED}
worktree       ${WT}
build-patch    $($BUILD_PATCH_APPLIED && sha1sum "$STEP_DIR/build-fix.patch" | cut -d' ' -f1 || echo "not needed")  build-fix.patch
cxxflags       $(python3 -c "
import json
for e in json.load(open('$WT/builddir/compile_commands.json')):
    if e['file'].endswith('plane_lot_dispatch.cpp'):
        print(' '.join(t for t in e['command'].split()
                       if t.startswith(('-O','-D','-m','-std=')) or t == '-g')); break
" 2>/dev/null || echo unknown)
PROV

sbatch "$STEP_DIR/vtune_job.sh" "$WT/builddir/bench" "$SAFE_REV" "$NUM_THREADS" "$ITERATIONS"
