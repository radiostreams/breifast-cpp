# Step 12: why scaling and CPU utilization are low

Node: 2 sockets x 36 cores, no SMT, 4 NUMA domains of 18 cores (SNC2), 54 MiB L3 per socket.
All runs: `bench` on `lot_diff36_cpp`, exclusive node. Weak scaling uses 15 items per thread.

## Scaling (`scaling-job66814019.out`)

| threads | strong, 1080 items | speedup | weak | weak efficiency |
|---:|---:|---:|---:|---:|
| 1 | 383.0 s | 1.0x | 5.31 s | 100% |
| 2 | 193.6 s | 2.0x | 5.57 s | 95% |
| 4 | 99.3 s | 3.9x | 5.65 s | 94% |
| 9 | 46.0 s | 8.3x | 6.23 s | 85% |
| 18 | 33.0 s | 11.6x | 8.54 s | 62% |
| 36 | 16.2 s | 23.6x | 8.83 s | 60% |
| 72 | 9.9 s | 38.8x | 10.30 s | 52% |

Idle time and load imbalance explain little of this. Each item gets slower under concurrency:
`process_band` takes 395 ms alone and 698 ms at 72 threads (1.77x, step 11 Tracy trace). The
slowdown per zone follows memory intensity: `band_accumulate` 2.34x, `combine` 2.11x, `median_of`
1.95x, `compute_rms_noise_map` 1.86x, `median_of_abs` 1.47x, `compute_snr_and_apply_filters`
1.35x. `label_islands`, which allocates 3 x 75 MB per call, slows down 3.39x.

## Isolating the cause (`numa-isolation-66840077.out`, `counters-66840544.out`)

`bench` now prints `getrusage` and `/proc/vmstat` deltas per iteration. The single-arena binary is
`builddir/bench` copied to a directory without `libtbbbind`, so TBB sees one NUMA node.

| run | wall | CPU util | user CPU | kernel CPU |
|---|---:|---:|---:|---:|
| T=1, domain 0 | 5.23 s | 99.8% | 5.2 s | 0.1 s |
| T=18, domain 0, local memory | 8.10 s | 94-97% | 136 s | 3.0 s |
| T=18, domain 0 cores, memory interleaved over 4 domains | 6.30 s | 94-95% | 105 s | 2.3 s |
| T=72, single arena | 10.08 s | 91% | 623-664 s | 14-21 s |
| T=72, 4 NUMA arenas | 10.41 s | 76-81% | 574-579 s | 12-13 s |

Bound to one domain with local memory, weak efficiency already drops to about 65% at 18 threads,
so remote memory plays no part in the main drop.

## Causes of low scaling

1. **DRAM bandwidth per NUMA domain.** Interleaving the same 18 threads' memory over all 4 domains
   cuts their time by 22%, although 3/4 of accesses become remote. Memory stalls count as user CPU
   time (136 s vs 105 s for identical work), so utilization stays high while items slow down.
2. **Filling all 4 domains** adds about 1.25x on top (10.1 s at 72 threads vs 8.1 s for one full
   domain). Input data is first-touched by the main thread on a single domain, and autoNUMA runs.
3. **Memory churn.** Each iteration faults and zeroes about 36 GB (18K huge-page faults): scratch is
   recreated on every `run_batch_core` call (72 x ~270 MB) and `label_islands` allocates 3 x 75 MB
   per call. Kernel time is 2-3% of CPU. Each thread's first item per iteration costs ~250 ms extra.
4. **autoNUMA** (`numa_balancing=1`) migrated 3.7M pages (~14 GB) during one 72-thread iteration.

## Cause of low CPU utilization

VTune's 78% comes from the 4-NUMA-arena mode. Its fixed per-domain item split cannot steal across
domains. It uses about 65 s less user CPU than the single arena thanks to locality, then loses that
to idle time, so wall time ends up equal. The single arena runs at 91%.

## Notes on earlier measurements

- Every VTune run in step 11 loaded `hwloc`, so it ran 4 NUMA arenas. The step 11 Tracy run used a
  single arena (`builddir-tracy` has the placeholder `libtbbbind`). That mode difference accounts
  for most of the 95% (Tracy) vs 78% (VTune) gap.
- The ITT pause/resume calls in `bench.cpp` duplicate the ones already in `run_batch_core`.

## Persistent scratch (`counters-66847371.out`)

- `scratches` in `run_batch_core` is now a `static tbb::enumerable_thread_specific`, so each thread
  keeps its buffers across calls.
- `label_islands` fills a caller-owned `IslandLabels`, which lives in `ProcessBandScratch` and holds
  its working buffers (`parents`, `root_to_compact_id`). `peak_per_label` is a vector indexed by
  `label - 1`, which also gives a deterministic detection order. The background is zeroed during
  pass 1, and only the labels actually created get initialized.
- Tests: `test_run_batch_core_scratch_persistence.cpp` checks results after a dirty previous call
  and after plane-shape changes against a fresh `ProcessBandScratch`. Removing the histogram reset,
  the `snr` resize or `peaks.clear()` makes them fail.

Iteration 1 (buffers already allocated), vs `counters-66840544.out`:

| run | wall | kernel CPU | huge-page faults | minor faults |
|---|---:|---:|---:|---:|
| T=1, domain 0 | 5.23 -> 5.18 s | 0.1 -> 0.0 s | 233 -> 1 | 7.8K -> 1.3K |
| T=18, domain 0, local | 7.99 -> 7.77 s | 3.0 -> 0.1 s | 4508 -> 58 | 97K -> 2.5K |
| T=18, interleaved | 6.28 -> 6.24 s | 2.2 -> 0.1 s | 4410 -> 232 | 98K -> 9.7K |
| T=72, single arena | 9.78 -> 10.37 s | 21.3 -> 0.8 s | 18092 -> 872 | 977K -> 35K |
| T=72, 4 NUMA arenas | 10.08 -> 10.48 s | 12.7 -> 3.1 s | 17705 -> 872 | 610K -> 129K |

Memory churn is gone and one full domain runs about 3% faster. At 72 threads wall time shows no
measurable change: the two jobs ran on different nodes (jst037, jst247), and autoNUMA migrated
3.7M pages in the baseline's fast iteration and almost none here, a spread larger than the effect.
User CPU at 72 threads rose (623 -> 708 s): the time freed from the kernel went to memory stalls,
as expected when DRAM bandwidth is the limit.

## Radix-select medians (`counters-66850675.out`)

`std::nth_element` reorders its input, and all three callers still need their array afterwards, so
each median copied the full 37 MB array and then partitioned the copy (about 200 MB of traffic per
call, much of it writes). This has been the case since the first version: `xt::median` copies
internally.

`radix_median_of`/`radix_median_of_abs` (`src/median.hpp`) compute the same exact order statistic
in two read-only passes. Floats map to order-preserving `uint32` keys (sign-flip transform; for
abs, the sign bit masked off). Pass 1 histograms the top 16 key bits to find the bin holding rank
n/2, and pass 2 histograms the low 16 bits inside that bin. For even n, rank n/2 - 1 comes from the
same scan or from the largest key below the bin. The 256 KB histogram is `thread_local`.

- Single thread, 3072 x 3072: 113 -> 31 ms (abs), 81 -> 31 ms (signed).
- Tests (`test_median.cpp`): the hand-checked, numpy and edge cases (ties, zeros of both signs,
  subnormals, infinities) run against both implementations, plus 6000 random arrays where radix
  and `nth_element` must agree.

Mean of 2 iterations, vs `counters-66847371.out`:

| run | before | radix | change |
|---|---:|---:|---:|
| T=1, domain 0 | 5.20 s | 4.44 s | -15% |
| T=18, domain 0, local | 7.91 s | 5.89 s | -26% |
| T=18, interleaved | 6.19 s | 5.05 s | -18% |
| T=18, unbound | 7.70 s | 5.92 s | -23% |
| T=72, single arena | 10.41 s | 7.84 s | -25% |
| T=72, 4 NUMA arenas | 10.58 s | 8.48 s | -20% |

The gain grows with the number of threads sharing a domain, as expected when DRAM traffic is the
limit. Weak efficiency inside one domain rises from 66% to 75%, and at 72 threads from about 50%
to 57%. User CPU at 72 threads drops from 708 s to 519 s for the same work. The run landed on a
third node (jst203); the node-to-node spread seen so far is about 5%.

## Next

- Fuse pass 1 of each median into the loop that produces its array (`band_accumulate`,
  `compute_rms_noise_map`, `combine`), removing one full read per median.
- Boost scans the full label image once per island; per-island pixel lists would remove that.
- Count the remaining full-array passes per band and fuse where possible.
- NUMA-local data and scratch, keeping work stealing across domains.

## Reproducing

```
cp builddir/bench builddir/libtbb.so.12 profiling/bench-single-arena/
cp builddir/bench builddir/libtbb.so.12 profiling/bench-single-arena-v2/
sbatch optimization/step_12/run_scaling.sh
sbatch optimization/step_12/run_numa_isolation.sh
sbatch optimization/step_12/run_counters.sh
```
