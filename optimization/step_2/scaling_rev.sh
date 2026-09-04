#!/bin/bash
#
# Build one revision of this repo in a throwaway worktree and submit a thread-scaling sweep
# against it.
#
#   ./scaling_rev.sh <git-rev> <label> [iterations] [threads]
#   ./scaling_rev.sh perf-step_1-before before 1 "1 2 4 8 16 32 72"
#
# Same worktree + build-fix.patch approach as profile_rev.sh (see its header for why the patch
# exists: no commit in this repo configures on its own). Unlike profile_rev.sh, this forces
# buildtype=release and debug=true on every revision it builds, so revisions compared by this
# script share the same build flags -- this is a "did the algorithmic change help" scaling
# test, not the flags-isolated P1/P2 experiment profile_rev.sh is set up for.

set -euo pipefail

REPO=/work/ska/hchouh/src/breifast-cpp
STEP_DIR="$REPO/optimization/step_2"
WORKTREES=/work/ska/hchouh/src/breifast-cpp-worktrees

REV="${1:?git rev required}"
LABEL="${2:?label required}"
ITERATIONS="${3:-1}"
THREADS="${4:-1 2 4 8 16 32 72}"

SAFE_REV="${REV//\//_}"
WT="$WORKTREES/$SAFE_REV"

# A stale worktree would silently sweep the wrong source, so always rebuild it from scratch.
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

# The dev shell comes from the main checkout's flake: the worktree is dirty from the patch,
# and this keeps one definition of the toolchain across every revision swept.
nix develop "$REPO" --command meson setup "$WT/builddir" "$WT" -Dbuildtype=release -Ddebug=true
nix develop "$REPO" --command ninja -C "$WT/builddir" bench

# Read back by scaling_job.sh and copied into the result's provenance file.
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

sbatch "$STEP_DIR/scaling_job.sh" "$WT/builddir/bench" "$LABEL" \
  "/home/chouh/work/src/breifast/small_test/lot_diff36_cpp/" "$ITERATIONS" "$THREADS"
