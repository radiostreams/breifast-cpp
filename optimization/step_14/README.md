# Step 14: CPU assessment before a GPU port

Question: is the CPU implementation close enough to the hardware limits to call it done? Code: step 13 (uncommitted
since `d876096`), bench lot (1080 planes), single arena, data on `/scratch`. Two builds of the same code:

- **current:** `builddir`, meson `debugoptimized`: `-O2 -g`, no `-march`, `_GLIBCXX_ASSERTIONS=1`. 0 AVX instructions.
- **release:** `builddir-release`: `-O3 -g -march=icelake-server -fno-math-errno -DNDEBUG`. 74/74 tests pass.

Node: 2 x Xeon 8360Y (Ice Lake, 36 cores each, AVX-512), 4 SNC domains. Scripts: `run_cpu_assessment.sh`
(scaling + VTune), `run_stream.sh`.

## Scaling (`scaling-current-66905300.out`, `scaling-release-66905301.out`)

Second iteration of each run. Strong: 1080 items. Weak: 15 items per thread. CPU util and user time from getrusage.

| T | current strong | eff | CPU util | user | release strong | eff | user | current weak eff | release weak eff |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 309.05 s | 100% | 99.8% | 308 s | 249.09 s | 100% | 248 s | 100% | 100% |
| 2 | 155.27 s | 99.5% | 99.8% | 310 s | 125.04 s | 99.6% | 249 s | 96.2% | 95.0% |
| 4 | 78.65 s | 98.2% | 99.6% | 310 s | 63.88 s | 97.5% | 250 s | 94.3% | 93.8% |
| 9 | 35.46 s | 96.8% | 99.4% | 313 s | 28.53 s | 97.0% | 253 s | 92.3% | 86.5% |
| 18 | 17.79 s | 96.5% | 98.8% | 316 s | 14.88 s | 93.0% | 262 s | 84.6% | 89.0% |
| 36 | 9.38 s | 91.5% | 95.8% | 321 s | 8.47 s | 81.7% | 290 s | 84.6% | 73.9% |
| 72 | 6.56 s | 65.5% | 93.5% | 438 s | 5.81 s | 59.5% | 393 s | 65.9% | 65.4% |

- Threads stay busy (93-94% CPU at 72). The loss at 72 is CPU time per item: +42% (current) and +58% (release)
  over T=1, while it stays within 5% of T=1 up to 36 threads (current).
- Release is 19% faster on one thread and 11% faster at 72 (different nodes). Its CPU time per item grows more at
  72 threads (+58% vs +42%).
- Weak runs are 15 items per thread (4-6 s), so one slow item (0.41 s vs 0.28 s median) weighs more than in strong.

## Memory bandwidth (`stream-66906591.out`, `*-hpc-performance-summary.txt`)

STREAM 5.10 (OpenMP, 800 MB arrays): triad 300 GB/s for the node, 76 GB/s for one domain (18 threads, local).
STREAM does not count write-allocate traffic, so the DRAM traffic behind triad is about 400 GB/s node-wide. VTune
counts DRAM traffic, so 400 GB/s is the ceiling for its numbers. VTune's "Platform Maximum" (50 GB/s) is wrong on
these nodes, so its "DRAM Bandwidth Bound" percentage is meaningless.

| 72 threads | current | release |
|---|---:|---:|
| DRAM average | 133 GB/s | 147 GB/s |
| DRAM observed maximum | 297 GB/s | 363 GB/s |
| per-package average | 73 GB/s | 80 GB/s |
| remote NUMA accesses | 53% | 53% |
| Memory Bound (pipeline slots) | 36% | 66% |

Average traffic is a third of the ceiling; peaks reach 75-90% of it. Half the accesses are remote: the single
arena takes items on any core, while buffers stay on the node that first touched them.

## Vectorization (`*-hpc-performance-summary.txt`, `*-functions.csv`)

| 72 threads | current | release |
|---|---:|---:|
| FP share of uops | 6.9% | 6.4% |
| packed FP | 0.0% | 10.8% (SP 7.1%, DP 19.3%, all 256-bit) |
| SP + DP GFLOPS | 39.7 | 42.9 |
| CPI | 0.365 | 0.518 |

93% of uops are non-FP: compares, masks, key conversions, histogram increments, index arithmetic. FLOP counts
therefore understate the work, and the FP roofline says little about this code.

Top loops, release build, share of CPU time:

| loop | CPU | vectorized | memory bound | CPI |
|---|---:|---|---:|---:|
| `compute_snr_and_apply_filters:189` | 18.7% | no (scalar, AVX-128 encoding) | 6% | 0.28 |
| radix `select` pass 1 (`median.hpp:101`, 2 instances) | 11.0% | no | 26-70% | 0.6-0.9 |
| radix `select` pass 2 (`median.hpp:117`, 2 instances) | 9.4% | no | 12-55% | 0.4-0.5 |
| `process_band:393` | 7.2% | AVX-256 + FMA | 100% | 8.3 |
| `process_band:349` | 7.2% | no | 100% | 2.9 |
| `compute_rms_noise_map:99, 133, 145, 152` | 14.6% | 133, 145, 152: AVX-256 | 38-100% | 0.5-2.2 |

- `compute_snr_and_apply_filters:189` is the largest loop and is compute bound (6% memory bound, CPI 0.28). It
  stays scalar at `-O3 -march` because the elementwise SNR and mask work shares the loop with the histogram
  scatter and the min/max reduction.
- The radix histogram loops are scatters, which do not vectorize.
- The current build shows `std::vector::operator[]` at 4.2% of CPU: `_GLIBCXX_ASSERTIONS` bounds checks in the
  radix histogram.

## Stalls (`*-uarch-exploration-summary.txt`)

Unusable. VTune runs in driverless perf mode here, the top-down counters are multiplexed, and level 1 does not add
up: Retiring + Front-End + Bad Speculation + Back-End = 137% (current), 152% (release). The per-loop Memory Bound
column of `hpc-performance` above is the stall data to use.

VTune's Effective CPU Utilization (66% current, 62% release) disagrees with getrusage (93-94%) and attributes no
spin or overhead time, so it is left out.

## Assessment

The threading is sound: 93-94% CPU busy at 72 threads, 92-97% strong-scaling efficiency up to 36. The code is at
neither hardware limit: scalar on an AVX-512 part, one third of DRAM bandwidth on average, half the accesses remote.
Measured headroom, in order of evidence:

1. **Build flags.** Release build: -19% on one thread, -11% at 72. No code change.
2. **`compute_snr_and_apply_filters:189`.** 19% of CPU, compute bound, scalar. Splitting the elementwise part from
   the histogram scatter lets it vectorize.
3. **NUMA locality.** 53% remote accesses at 72 threads.
4. **Remaining full-array passes** (step 13 #4, #5): each pass removed in step 13 cut 72-thread time by 3-14%.

Radix histograms (20% of CPU) remain scalar by nature.

## #1 done: build flags

`meson.build` defaults to `buildtype=release`, `debug=true` (symbols for VTune), `b_ndebug=true`, and adds
`-march=<march> -fno-math-errno` through `add_project_arguments`. `march` is a meson option, default `native`.
The wheel builds `release` (`pyproject.toml`). Every build dir was wiped and set up again with these defaults.

TBB builds `libtbbbind` only when hwloc is visible to `/usr/bin/pkg-config`, so builds need
`module load hwloc/2.9.1` before `nix develop`. Without it `libtbbbind` is an empty placeholder and TBB runs a single
arena.

VTune hotspots, hpc-performance and memory-access of this build (72 threads, single arena, before #2):
`profiling/vtune-native-threads72-job66907416/` (`run_vtune_trio.sh`, not analyzed here).

## #2 done: `compute_snr_and_apply_filters` with xsimd (`ab-66909635.out`)

xsimd 14.3.0 is a dependency (`flake.nix`, `meson.build`). The loop processes one `xsimd::batch<float>` (16 lanes on
AVX-512) per step: SNR, the four masks as `batch_bool<float>`, the bin index, the min/max as batch accumulators
reduced after the loop, and the final threshold mask. The histogram increments stay scalar: the bin indices are
stored to a 16-entry array and each histogram walks the set bits of its mask. The scalar loop, unchanged, is the
epilog for the pixels past the last whole batch (all of them with `SCALAR` defined).

Same-node A/B (jst034), before = the same `-O3 -march=native` build with the scalar loop:

| config | before | after | change |
|---|---:|---:|---:|
| one thread, per item (mean of 36) | 0.2307 s | 0.1772 s | -23.2% |
| T=18, domain 0, local | 4.386 +- 0.047 s | 3.944 +- 0.033 s | -10.1% |
| T=72, single arena | 5.435 +- 0.141 s | 4.770 +- 0.203 s | -12.2% |

Same detections (36, 150). The gain shrinks with thread count: the loop was compute bound, and at 72 threads the
per-item cost is set by DRAM traffic, which this change leaves unchanged. The remaining gains at scale are items 3
and 4 above.

Against earlier runs on other nodes (+-5%), T=72 is now about 5.2x the first C++ version (24.89 s) and about 21x
Python on the bench lot (102.35 s).
