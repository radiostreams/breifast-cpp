# Step 8 -- where the sliding-window rewrite stands, and where the time actually goes

Status: step 7's ring-buffer rewrite of `compute_rms_noise_map` is uncommitted WIP. This step
measures it against `main`, and adds Tracy zone instrumentation to find the time step 1-7 never
looked at.

## 0. Headline: first C++ vs last, and Python vs last

First C++ (`1f30b6e`) vs last (main, `72be5f7`), same bench/manifest as section 1: **24.888 s ->
11.110 s, 2.24x**.

Python vs last C++, real pipeline (`tron-pfb`, `test.ms`+`deep-image.fits`, `stokes: IQUV`, 72
threads, jobs 66642585/66642586): 440 `process_plane`/`process_lot` calls each.

| | wall/compute time | per item |
|---|---:|---:|
| Python | 43.41 s | 98.7 ms |
| C++ (main) | 12.82 s | 29.1 ms |

**3.4x**, on real pipeline data (section 1 uses a synthetic replicated manifest).

## 1. Timing: initial vs main vs WIP

`bench <manifest> 72 5`, all three binaries in one allocation (job 66649989, jst233), all built
`debugoptimized` (`-O2`, `debug=true`).

| | revision | mean iteration | vs initial | per-item, single-threaded |
|---|---|---:|---:|---:|
| initial | `1f30b6e` | 24.888 s | 1.00x | 0.6125 s |
| previous | `72be5f7` (main) | 11.110 s | **2.24x** | 0.3700 s |
| current | main + WIP | 11.423 s | 2.18x | 0.3960 s |

Step 7's headline reproduces: it recorded 24.768 s / 10.926 s (2.27x), today 2.24x.

**The WIP is a regression.** Wall-clock +2.8% sits inside the noise (11.110 +/- 0.380 vs
11.423 +/- 0.363, difference 0.313 +/- 0.235 at n=5). Per-item +7.0% is real: main's p75 (0.3736)
is below the WIP's p25 (0.3819), so the middle halves do not overlap.

## 2. Memory access: did the rewrite do what it was for?

`vtune -collect memory-access`, 72 threads, 10 iters, identical hardware (Icelake, 72 CPUs,
2.394 GHz, 56.6 MB LLC). main = job 66650508 (jbm015), WIP = job 66650272 (jbm024).

| metric | main | WIP | change |
|---|---:|---:|---:|
| Memory Bound | 100.0% | 78.2% | -21.8 pts |
| L1 Bound | 10.2% | 8.1% | -2.1 pts |
| L2 Bound | 19.7% | 18.8% | -0.9 pts |
| DRAM Bound | 12.6% | 11.3% | -1.3 pts |
| LLC Miss Count | 2.407 B | 2.192 B | -8.9% |
| Remote Memory Access | 899.0 M | 710.8 M | -20.9% |
| Remote Cache Access | 164.3 M | 59.8 M | -63.6% |
| NUMA remote | 42.0% | 37.3% | -4.7 pts |
| UPI Utilization Bound | 62.7% | 54.6% | -8.1 pts |
| **Loads** | **3.645 T** | **4.698 T** | **+28.9%** |
| **Stores** | **1.058 T** | **1.233 T** | **+16.5%** |
| CPU Time | 4771.7 s | 5129.8 s | +7.5% |
| DRAM avg | 129.4 GB/s | 127.1 GB/s | -1.8% |
| DRAM high-BW time | 67.4% | 68.7% | +1.3 pts |

**The mechanism works and does not pay.** Every locality metric improves. Total DRAM traffic does
not move (both ~72% of the 178 GB/s ceiling), and +28.9% loads costs +7.5% CPU time, which matches
the +7.0% per-item regression. Closing the load gap is what stands between this approach and
break-even.

Caveat: main's Memory Bound reads exactly 100.0%, the metric pegged at its ceiling, so -21.8 pts
likely overstates. Direction is safe, since LLC misses, remote accesses and UPI move independently.

## 3. Tracy instrumentation

Tracy 0.14.2 (`81c50055`) in `tracy/`, gitignored. Build option `-Dtracy=true` (see `meson.options`
and `meson.build`): `bench` gets `TracyClient.cpp` plus `TRACY_ENABLE/NO_EXIT/ONLY_LOCALHOST/
NO_BROADCAST`, the wheel and Catch2 tests get the headers only. With the option off, `bench`
contains zero Tracy symbols.

`TRACY_NO_EXIT` makes the process block at exit until `tracy-capture` drains the trace, which is
required in a batch job and means the binary hangs at exit if run without a capture alongside it.

### Zone breakdown

Job 66651155, 72 threads, 3 iters, 33,579 zones. Exclusive time per `process_band` call
(mean 832.6 ms). `pad_image`/`label_islands`/`detections` run on 273 of 3276 calls, since most
planes have nothing surviving the SNR filters, so they are amortized across all calls.

| phase | exclusive | share |
|---|---:|---:|
| `compute_rms_noise_map` own work | 157.1 ms | 18.9% |
| **`median_of_abs`** (plane) | **141.7 ms** | **17.0%** |
| `compute_snr_and_apply_filters` | 146.4 ms | 17.6% |
| `combine` own work | 79.6 ms | 9.6% |
| **`median_of`** (windows_sums, in rms map) | **74.1 ms** | **8.9%** |
| **`median_of`** (noisemap, in combine) | **74.1 ms** | **8.9%** |
| `band_accumulate` | 69.2 ms | 8.3% |
| `label_islands` | 21.1 ms | 2.5% |
| `mask_accumulate` | 11.6 ms | 1.4% |
| `detections` | 8.6 ms | 1.0% |
| `pad_image` | 5.1 ms | 0.6% |
| unattributed | 43.9 ms | 5.3% |

**Medians are 34.8% of per-item time**, across three call sites. That exceeds
`compute_rms_noise_map`'s own work. Steps 1-7 optimized the 45% that the first two zones covered.

`median_of`/`median_of_abs` both `scratch.assign()` the whole plane (9.4 M floats, ~37 MB) before
`nth_element`, three times per plane, so ~113 MB of copy traffic per plane feeding a noise
estimate. Step 3 already cut these once (`xt::median` -> `std::nth_element`, -18.4%); the copy and
full partition survived.

## Next

1. Histogram-based or subsampled median. Single pass, no copy, no sort. Targets ~35% of runtime.
2. Close the WIP's +28.9% load gap, which is worth the ~7% it currently costs.
3. Decoupled NUMA pinning (one arena + `task_scheduler_observer` + `sched_setaffinity`), to settle
   step 5's confounded result. Low expected value: today's data shows remote-to-local conversion
   leaves total DRAM traffic unchanged.

## VTune collector reliability

9 `memory-access` attempts, 3 usable. Failures show `tmu.c:416` or `ittcol.c:357` assertions plus
`Data file is corrupted`, and are only detectable by checking that the target actually ran.
A good capture has CPU Time ~5100-5500 s against ~270-280 s elapsed and prints `loaded 1080
planes`; every failure sat under 55 s CPU with no program output. Node-independent (jbm024 both
failed and succeeded). Consistent with step 1 and step 2's notes.

## Reproducing

```bash
sbatch run_bench_compare3.sh 5 72                 # three-way timing
sbatch run_bench_memaccess.sh <exe> <label> 72 10 # memory-access, one binary per job
sbatch run_bench_tracy.sh 72 3                    # tracy capture

nix develop --command meson setup builddir-tracy -Dtracy=true
nix develop --command ninja -C builddir-tracy bench
cmake -S tracy/capture -B tracy/capture/build -DCMAKE_BUILD_TYPE=Release && cmake --build tracy/capture/build -j
cmake -S tracy/csvexport -B tracy/csvexport/build -DCMAKE_BUILD_TYPE=Release && cmake --build tracy/csvexport/build -j
tracy/csvexport/build/tracy-csvexport profiling/bench-threads72-iters3-job66651155.tracy
```

Traces and VTune results are under `profiling/` (gitignored).
