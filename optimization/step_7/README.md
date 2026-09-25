Status:
After step 6, the main issue remaning is memory access patterns.
Looking at a VTune profiling  and analyzing it with Claude, it appears that the main issue seems to be compute_rms_noise_map's summed-area table that has memory access patterns that give guaranteed cache misses.

Hypothesis:
Switching to a 1D sliding window algorithm would reduce cache pressure and improve memory access patterns and cache line locality.

Action:
First implementation didn't go well, 26% slower. Profiling.

## Bug found during review

The ring-buffer rewrite of `compute_rms_noise_map`'s summed-area table (`BOX_VERSION` branch,
`src/plane_lot_dispatch.cpp`) had a use-after-free: `parts_sums.data()` was captured *before*
`parts_sums.resize(...)`, which reallocates whenever the shape changes -- the captured pointer
then pointed at freed memory. Intermittent crash, not deterministic. Fixed by taking the pointer
after `resize()`. A second bug in the same rewrite (a single flat loop counter indexing both
`plane`, row width `n_x`, and `parts_sums`, row width `n_x+1`) was also caught and fixed before
it shipped -- see the fixed code for the corrected per-row offset math.

## `strip_width` sweep

The separable box filter (the `#else`/production branch) processes columns in strips so the
per-row ring buffer stays cache-resident; `strip_width` was hardcoded to 256, chosen only for
single-thread L2 residency (comment: "256 keeps the ring under 1 MB for boxsize 400"), never
checked against real multi-threaded contention. Swept 64..3072 (`run_strip_width_sweep.sh`, job
66584967, 72 threads, 5 iters, current `main`):

| strip_width | mean (s) | per-item single-thread (s) |
|---:|---:|---:|
| 64 | 13.591 | 0.5395 |
| 128 | 11.669 | 0.4669 |
| 256 (old default) | 11.482 | 0.4294 |
| 512 | 11.213 | 0.4108 |
| 1024 | 10.921 | 0.4006 |
| **2048** | **10.752** | 0.3968 |
| 3072 (no stripping) | 10.988 | 0.3933 |

2048 wins on wall-clock (-6.4% vs 256), even though 3072 has the best single-threaded per-item
cost -- at 72 threads the full-width ring per thread pressures shared L3 enough to lose to 2048
under real concurrency. Set as the new default.

## Result

Headline before/after (`bench <manifest> 72 5`, job 66585621): `before` = `perf-step_1-before`
(`1f30b6e`, first working `xt::xarray` version), `after` = current `main` with the box-filter
rewrite + `strip_width=2048`.

| | before | after |
|---|---:|---:|
| mean iteration (`run_batch_core`) | 24.768 s | 10.926 s (**-55.9%, 2.27x**) |
| per-item cost, single-threaded (mean) | 0.6148 s | 0.3964 s (-35.5%) |

## Reproducing

```bash
sbatch run_strip_width_sweep.sh          # strip_width sweep, current main
sbatch run_bench_compare.sh /work/ska/hchouh/src/breifast-cpp-worktrees/perf-step_1-before/builddir/bench \
                             /home/chouh/work/src/breifast-cpp/builddir/bench
```