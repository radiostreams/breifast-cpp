# Optimization step 1 — `xt::xarray` → `xt::xtensor<T, N>`

| | |
|---|---|
| before | `perf-step_1-before` (`1f30b6e`, "Debug + testing + bench") |
| after | `perf-step_2-after` (`568bfcc`, "Performance optimization (scratch + statically structured arrays"; was `d7eee44` before an amend) |
| baseline measurement | VTune job 66340861, 2026-09-03 — **14.83 s/iteration** (5 iters) |
| baseline re-run from the tag | VTune job **66347278**, 2026-09-04 — **19.76 s/iteration** (1 iter, no `-start-paused` — see "Known issues") |
| build flags | `debugoptimized` on **both** revisions, deliberately — see "Scope" |

Raw reports (summary + per-run hotspots CSV, program log, provenance) for both baseline runs
are in `results/`.

## Known issues

- **`1f30b6e` (the `perf-step_1-before` tag) has no ITT instrumentation.**
  `git show 1f30b6e:src/plane_lot_dispatch.cpp | grep itt` finds nothing — `__itt_resume`/
  `__itt_pause` were added later, uncommitted, sometime before job 66340861 ran but after
  this commit. `vtune_job.sh` used `-start-paused`, which waits for the target to call
  `__itt_resume()` before collecting anything; since the tagged binary never does, the
  collector stayed paused for the entire run. Two attempts (66343033, then 66347252) both
  produced an empty ~6 MB capture (zero hotspot rows, no `CPU Time`) before this was
  diagnosed. **Fixed**: `vtune_job.sh` now checks `nm "$EXE" | grep __itt_resume_ptr` and
  only adds `-start-paused` if the binary actually supports it, printing a warning
  otherwise. This means a from-tag capture profiles the *whole process* (including the
  manifest load and the single-threaded per-item timing loop), not just the parallel
  region — a broader window than 66340861's paused capture.
- **Without `-start-paused`, this cluster's collector corrupts on longer runs.** With the
  fix applied but still at `iterations=5` (~220s full-process elapsed), two more attempts
  (66347270, 66347275) both failed identically: `Data file is corrupted`, cut off at
  ~50s elapsed, ~7.3-7.6 MB. Dropping to `iterations=1` (~69s elapsed) succeeded cleanly
  (job 66347278, 51 MB, real hotspot data, and its `xt::median` call site correctly shows
  `xt::xarray_container` — confirming it really is the pre-migration binary). Looks like a
  duration or data-volume limit on the driverless collector for full-process captures;
  not otherwise investigated. **If you need a multi-iteration from-tag capture, run
  several separate single-iteration jobs instead of one 5-iteration job.**
- **Run-to-run wall-time variance is large.** Three separate executions of nominally the
  same `perf-step_1-before` code gave three different numbers: 14.834 s/iter (66340861,
  dirty pre-tag tree, paused capture), 25.019/25.073 s/iter (66343033's and 66347252's
  `bench` logs — both had valid program timing even though their VTune capture failed),
  and 19.759 s/iter (66347278, clean tag, unpaused capture). All ran on different nodes
  (jst163, jst012, jst021...). Treat any single iteration count here as noisy; the
  "14.83 s/iteration" headline this document leads with has not been reproduced.

## What the profile said

Baseline hotspots (full analysis: `notes/analyses/2026-09-03-baseline-vtune-job66340861.md`),
1080 plane items × 5 iterations, 3072² f32 planes, 72 threads:

| category | CPU time | instructions | CPI |
|---|---|---|---|
| xtensor index/stepper machinery | 42.8 % | **56.8 %** | 0.26 |
| median / `nth_element` / partition | 19.2 % | 10.2 % | 0.66 |
| user code (self) | 17.9 % | 23.4 % | 0.27 |

56.8 % of all retired instructions were address arithmetic. `xt::xarray` has *dynamic rank*:
shape and strides live in an `xt::svector`, so `a(i_y, i_x)` cannot be hoisted or
strength-reduced. The inner loop of the summed-area table carried a runtime rank check and
two integer multiplies per element:

```
 290: cmp $0x10,%rsi     <- "strides.size() == 2", re-tested every pixel
 29a: mov (%r12),%rcx    <- strides reloaded from the svector
 2a6: imul %r9,%r14
 2b0: imul %r15,%r10
```

Parallelism is good: 87.6 % effective CPU utilization, zero spin, zero
overhead.

## What changed

Every container moved from `xt::xarray<T>` to `xt::xtensor<T, N>`, which stores shape and
strides in a `std::array<size_t, N>` and unrolls the offset at compile time. Touched
`ProcessBandScratch`, `PlaneBatchItem`, `PlaneBatchGroup`, `PlaneConfig`, `VarianceStats`,
`IslandLabels`, the pybind11 boundary, and the bench loader. `src/` now contains no
`xt::xarray`.

Two things were considered and rejected:

- **Hand-rolled `Span2D`/`Buffer2D` wrappers over `std::vector`.** Proposed by the AI
  assistant, rejected: `xt::xtensor<T, N>` is the library's own static-rank container and
  `.data()` / `&a(y, 0)` is the native escape hatch for the innermost loops. Reinventing an
  abstraction the library already provides.
- **Changing build flags at the same time.** Kept separate so the two effects can be
  measured independently (P1 vs P2 in `notes/analyses/INDEX.md`).

## What static rank exposed

The migration was not a mechanical type substitution. Fixing the rank surfaced defects that
`xarray` had been silently absorbing, because rank was a runtime property:

| defect | site | latent because |
|---|---|---|
| six struct fields declared at the wrong rank | `accept_mask`, `mean_image`, `flux_cutoff`, `snr_cutoff` (should be 3); `baseline_flux` (2); `beam_*_axis` (1) | any rank assigns to `xarray` |
| `labels_.resize({0})` on a rank-2 container | `islands.cpp` | resizing an `xarray` to a 1-D shape just changed its rank |
| `nonzero_channels_mask` rank 2 vs assigned rank 1 | `plane_lot_dispatch.hpp` / `.cpp` | as above |
| four bench loader call sites at rank 2 for rank-3 data | `bench.cpp` | `resize()` from a `std::vector` shape truncates into `std::array` |
| `plane` both rank 3 and rank 2 in one variable | dead `#if USE_XTENSOR` band collapse | no fixed-rank type can express it; the branch was deleted |

Worth noting for the write-up in `notes/ai_use_for_hpc.md`: **the compiler caught almost
none of this.** `plane_lot_dispatch.cpp` and `islands.cpp` built clean on the first attempt.
The only compile error in the whole migration was a template-arity mismatch in the bindings.
Every genuine rank defect was found by reading declarations against the dataset manifest.

## Verification

- `ninja` clean: zero errors, zero warnings
- Catch2 suite: **all tests passed (12469 assertions in 67 test cases)**
- pytest: blocked, unrelated — VTune's `libittnotify.a` contains `ittptmark64.pic.o` with no
  `.note.GNU-stack`, so the linker marks the extension module's stack `RWE` and the kernel
  refuses to load it. Fix is `-Wl,-z,noexecstack` on that target, or not linking ITT into the
  Python module at all (only `bench` needs it, for `-start-paused`).

## Reproducing

```bash
./profile_rev.sh perf-step_1-before      # baseline
./profile_rev.sh perf-step_2-after       # after the migration
```

`profile_rev.sh` checks the revision into a throwaway worktree under
`/work/ska/hchouh/src/breifast-cpp-worktrees/`, applies `build-fix.patch`, builds `bench`
with `nix develop`, and submits `vtune_job.sh`. Results land in `profiling/` with the
revision label in the directory name and a `provenance.txt` recording the resolved SHA, the
patch hash and the actual compiler flags.

### Why build-fix.patch exists

**No commit in this repository configures on its own.** `meson.build` references `itt_dep`,
`itt_root` and `tbbmalloc_dep` at lines 82/99/125, and none of them has ever been defined in
any commit — `git log --all -S 'itt_dep ='` returns nothing. The build has only ever worked
from a dirty working tree. The patch carries the minimum needed to configure: the `itt_dep`
declaration, dropping the undefined `tbbmalloc_dep`, the `tbb_build` `install_dir` arity, and
`nlohmann_json` in `flake.nix`.

It deliberately does **not** touch `buildtype`, so each revision is profiled with the flags
it was actually committed with. Folding these fixes into a real commit would make the patch
unnecessary for future steps; it cannot retroactively fix `1f30b6e`.

## Scope

This step changes container types only. Still open from the baseline analysis, in the order
they are worth doing:

- **P1** build flags — `-O2`, no `-march` (zero AVX emitted on an AVX-512 part),
  `_GLIBCXX_ASSERTIONS=1` still on. `buildtype=release` is in the working tree, uncommitted.
- **P3** `xt::median` → `std::nth_element` on `.data()` into reused scratch (19.2 %)
- **P4** separable box filter replacing the summed-area table (75 MB/thread, CPI 243)
- **P5** hoist `histogram_detail`, specialize the single-channel cutoff path, `snr` out-param
