# Step 15: remaining SIMD candidates

Starting point: commit `5c5804d` (step 14: release flags, xsimd `compute_snr_and_apply_filters`), T=72 single arena
about 4.7 s on the bench lot.

## Candidates (`vtune-xsimd-66909744-*`)

VTune hotspots, hpc-performance and memory-access of `5c5804d` (`run_vtune_trio.sh`, 72 threads, 3 iterations,
single arena, data on `/scratch`, jst143). Shares are of total CPU time in the hpc-performance per-loop report
(`vtune-xsimd-66909744-hpc-performance-functions.csv`, 976.6 s).

| # | loop | CPU | state | why | fix | effort |
|---|---|---:|---|---|---|---|
| 1 | radix `select` pass 2 (`median.hpp`, 2 instances) | 10.2% | scalar, CPI 0.4-0.5 | per element: key, branch on `high == bin`, else `max` | keys, compare and masked `max` on batches; scalar increments for the lanes in the bin | medium |
| 2 | radix `select` pass 1 (2 instances) | 12.5% | scalar, CPI 0.8-0.9 | histogram increment is a scatter | keys on batches, increments scalar | medium |
| 3 | `band_accumulate` (`process_band:442`) | 8.3% | scalar, CPI 3.2, 100% memory bound | 3-D `xtensor` `operator()`, `std::optional` deref and `do_subtract_mean` branch per element | raw pointers, branch hoisted | easy |
| 4 | `compute_rms_noise_map:100` (horizontal sliding sum) | 5.9% | scalar, CPI 0.55 | running `double` accumulator along each row | rows in parallel (8x8 transposes) or prefix sums | hard |
| 5 | `compute_rms_noise_map:134` (sqrt of window mean) | 4.9% | partly vectorized | `count` recomputed per element | column widths once per plane | easy |
| 6 | `compute_rms_noise_map:146/153` (column accumulators) | 5.6% | AVX-256, memory bound | two passes over `column_accumulators` | one loop, `acc += entering - leaving` | easy |
| 7 | `image_mask` copy + `xt::any` | 2.7% | partly vectorized | extra full passes over the mask | OR while copying, `any` returned by `compute_snr_and_apply_filters` | easy |
| 8 | `combine` (`process_band:486`) | 7.7% | AVX-256 + FMA, 40% DRAM bound | already vectorized | none | none |

The key helpers listed separately by VTune (`bit_cast`, `abs_key`, `signed_key`, 12%) are inlined in #1 and #2, which
puts the radix medians at about 35% of CPU. At 72 threads Memory Bound is 69-84%, so the memory-bound rows (3, 6,
7, 8) mainly help on one thread or by cutting traffic.

## #1 and #2 done: radix `select` on xsimd batches (`ab-66937994.out`)

`SignedKey` and `AbsKey` bundle the scalar key, the batch key and the inverse; `select<Key>` takes one of them.

- Pass 1: keys and `>> 16` on a batch, stored to a lane array, scalar increments.
- Pass 2: keys, `high < bin` masked `max` into a batch accumulator reduced after the loop; the lanes with
  `high == bin` (a few per batch at most) are incremented scalar, and batches with none skip that step.
- A scalar loop covers the values past the last whole batch.

`test_median.cpp` checks both medians against the `nth_element` reference.

Same-node A/B (`run_ab.sh`, jst240), before = `5c5804d`; summary from `ab_summary.py ab-66937994.out`:

| config | before | after | change |
|---|---:|---:|---:|
| one thread, per item (mean of 36) | 0.1770 s | 0.1462 s | -17.4% |
| T=18, domain 0, local | 3.927 +- 0.052 s | 3.733 +- 0.068 s | -4.9% |
| T=72, single arena | 4.694 +- 0.140 s | 4.404 +- 0.128 s | -6.2% |

Same detections (36, 150). The gain shrinks at scale: both passes read the full array, which stays the limit under
load.

Against earlier runs on other nodes (+-5%), T=72 is now about 5.7x the first C++ version (24.89 s) and about 23x
Python on the bench lot (102.35 s).

## Reproduce

```
# build (real libtbbbind needs hwloc)
module load hwloc/2.9.1 && nix develop --command ninja -C builddir bench test_breifast_core
./builddir/test_breifast_core

# A/B: <dir>/before and <dir>/after each hold bench + libtbb.so.12 (no libtbbbind: single arena)
sbatch run_ab.sh 5 <dir>
python3 ab_summary.py ../../profiling/ab-<jobid>.out

# VTune trio on one build dir holding bench + libtbb.so.12
sbatch run_vtune_trio.sh <exe dir> <label>
```
