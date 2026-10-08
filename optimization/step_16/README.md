# Step 16: `band_accumulate` (step 15 candidate #3) and lazy `noise_global_mad`

Starting point: step 15 working tree (radix `select` on xsimd batches, `ab-66937994.out`). Loop under test: the
`band_accumulate` zone in `process_band` (`src/plane_lot_dispatch.cpp`), 8.3% of CPU in the step 15 VTune report, scalar,
CPI 3.2, 100% memory bound.

## Result

| change | outcome |
|---|---|
| flat loop on raw pointers, shape guards, branches hoisted | kept. About -3% at one thread and at T=72 |
| tiling over pixels with the channel loop inside (tile 2048 floats) | dropped. No gain at one thread. The paired T=72 run on the 4-channel synthetic lot gives +0.4% (t = 0.08). The earlier favorable results (T=18 -6.0%, T=72 -2.7%, sweep -5% to -11%) were inside the noise |
| lazy `noise_global_mad` (`process_band`) | kept. One thread -5.4% to -6.0% on both lots, same detections. T=72 -0.9% (bench lot, not resolved) and -5.9% (synthetic lot, t = -2.50, 4 rounds) |

## Change 1: flat loop

`xtensor` `operator()` and the per-element `std::optional` and `do_subtract_mean` checks are gone. Shapes of `stokes_cube`
and `mean_image` are checked once against the plane (throws `std::runtime_error`). Per channel, `if constexpr` flags select
first-channel assign or accumulate, and mean subtraction on or off. The operation order per pixel is unchanged, so the
output is bit-identical. The unit tests pass (74 cases).

Same-node A/B on the bench lot (`ab-40141941.out`), before = step 15 working tree:

| config | before | after | change |
|---|---:|---:|---:|
| one thread, per item | 0.1463 s | 0.1419 s | -3.0% |
| T=18, domain 0 | 3.692 +- 0.037 s | 3.720 +- 0.072 s | +0.7% |
| T=72, single arena | 4.458 +- 0.295 s | 4.322 +- 0.084 s | -3.0% |

Detections identical (36, 150). The T=72 gain is about one standard deviation of the before run.

## Change 2: tiling

Pixel tiles of 2048 floats, channels in the inner loop, one write of the tile to `plane`. Same per-pixel operation order.
Idea: remove the `plane` read and write that each channel pass does.

| lot | config | flat | tiled | change |
|---|---|---:|---:|---:|
| bench lot, 1 channel (`ab-40143081.out`) | one thread | 0.1420 s | 0.1426 s | +0.4% |
| | T=18 | 3.709 +- 0.047 s | 3.654 +- 0.054 s | -1.5% |
| | T=72 | 4.470 +- 0.286 s | 4.540 +- 0.347 s | +1.6% |
| synthetic, 4 channels (`ab-40146936.out`) | one thread | 1.2704 s | 1.2769 s | +0.5% |
| | T=18 | 33.74 +- 0.73 s | 31.72 +- 0.50 s | -6.0% |
| | T=72 | 11.92 +- 1.40 s | 11.59 +- 0.72 s | -2.7% |

A single-channel band has no channel loop to fuse, so the first lot cannot show a gain. On the 4-channel lot only T=18
moves (about 2 sigma). Decision taken at that point: drop the tiling. Correction 7 below revisits it.

Flat loop against the original on the synthetic lot (`ab-40146937.out`): one thread 1.3212 s to 1.2727 s (-3.7%), T=72
12.06 +- 1.28 s to 11.75 +- 0.70 s (-2.6%, 12 samples each). This agrees with the -3% of change 1.

## Change 3: lazy `noise_global_mad`

`noise_global_mad` (`radix_median_of_abs` of the plane, two full passes) was computed for every band. Its consumers are
the non-RMS branch (`median_noise = combine(...)`) and, inside the detection loop, `noisemap0_at_peak` (non-RMS mode) and
`flux / noise_global_mad`. The detection loop runs only after the `!xt::any(image_mask)` early return, and `plane` is read-only
from the end of `band_accumulate` on (`compute_rms_noise_map` and `compute_snr_and_apply_filters` take it as `const &`).

Change: a `std::optional<float>` local to `process_band` plus a lambda `noise_global_mad()` that fills it on first use.
The three consumers call the lambda, so the median runs at most once per band, and not at all for bands that return
early in RMS mode. The value is the same expression on the same data, so results are bit-identical. Unit tests pass (74
cases).

Paired runs (`run_t72_paired.sh`, T=72 single arena, 4 rounds with alternating order, 6 iterations per invocation,
iteration 0 dropped). In these two logs the label `flat` is the eager median (before) and `tile` is the lazy median (after).

| lot | metric | eager | lazy | change |
|---|---|---:|---:|---:|
| bench lot (`t72-paired-40192404.out`) | one thread, per item (mean of 4 invocations) | 0.1424 s (sd 0.0004) | 0.1339 s (sd 0.0006) | -6.0% |
| | T=72, 20 iterations | 4.424 s | 4.383 s | -0.9% (paired t = -0.81) |
| | detections | 150 | 150 | |
| synthetic lot (`t72-paired-40192405.out`) | one thread, per item | 1.2707 s (sd 0.0086) | 1.2016 s (sd 0.0078) | -5.4% |
| | T=72, 20 iterations | 12.019 s | 11.309 s | -5.9% (paired t = -2.50, 3 of 4 rounds negative) |
| | detections | 0 | 0 | |

Reading:
- The one-thread gain is clear on both lots: the gap is 10 times the spread between invocations.
- At T=72 the bench lot shows -0.9%, inside the noise (paired standard error 0.05 s against a 4.4 s mean, so about 1%).
  The synthetic lot shows -5.9% with t = -2.50 over 4 rounds. That is short of the 5% threshold (t = 3.18 for 3 degrees of
  freedom). The T=72 gain is probable on the synthetic lot and undemonstrated on the bench lot.
- The synthetic lot has no detections, so every band skips the median. The bench lot has 150 detections at T=72 and its
  detections match, which checks the lazy path.

## Tile-size sweep

`run_tile_sweep.sh` (job 40189061, `tile-sweep-40189061.out`, summary from `tile_sweep_summary.py`). The tiled build reads the
tile size from `BAND_TILE` at run time. 4-channel synthetic lot, T=72 single arena, 3 iterations per invocation, 2 rounds
(second in reverse order), iteration 0 dropped, so n = 4 timed iterations per row. The one-thread column is the 36-item
single-thread sample that `bench` always runs.

| tile (floats) | one thread, per item | T=72 mean | vs flat |
|---|---:|---:|---:|
| flat | 1.2721 s | 13.42 +- 2.05 s | |
| 64 | 1.2481 s | 12.26 +- 1.56 s | -8.7% |
| 256 | 1.2603 s | 11.94 +- 1.41 s | -11.1% |
| 1024 | 1.2790 s | 12.00 +- 1.60 s | -10.6% |
| 2048 | 1.2690 s | 11.97 +- 1.19 s | -10.9% |
| 4096 | 1.2610 s | 12.14 +- 1.14 s | -9.6% |
| 8192 | 1.2643 s | 12.41 +- 1.46 s | -7.6% |
| 16384 | 1.2733 s | 12.74 +- 1.29 s | -5.1% |
| 32768 | 1.2627 s | 12.27 +- 0.68 s | -8.6% |

Reading:
- The T=72 means of the tile sizes span 11.94 s to 12.74 s with a standard deviation of 0.7 s to 2.0 s per row. The sweep
  cannot rank them. The tile size has no visible effect between 64 and 32768 floats.
- The single-thread column stays within +-2% of flat for every size.
- All eight tile sizes sit below the flat row, by 5% to 11%. That gap is larger than the -2.7% of job 40146936, because the
  flat row of this job is the slow one (13.42 s, against 11.92 s and 11.75 s in two earlier runs of the same flat binary).
  The tile rows (11.9 s to 12.7 s) are close to the 11.59 s of the earlier tiled run. With n = 4 and a spread of 1 to 2 s
  the flat-against-tiled difference is about one standard error.
- My guess in the memory profile section (a tile-sized chunk read at a large stride suits the prefetcher worse) predicts
  small tiles doing worse. Tile 64 is among the fastest, so the data gives no support for that guess.

Reading: the direction favors tiling again. The size of the gain is between 3% (paired same-node runs) and 9% (this sweep,
whose flat row is slow) and none of the individual comparisons reaches significance. The tile size does not matter.

## Paired T=72 run

`run_t72_paired.sh` (job 40191867, `t72-paired-40191867.out`, summary from `t72_paired_summary.py`). Flat against tile 2048,
4 rounds with alternating order, 6 iterations per invocation (iteration 0 dropped, 5 timed), so 20 timed iterations per
variant, T=72 single arena, 4-channel synthetic lot, one node (jbm004).

| | flat | tiled |
|---|---:|---:|
| mean over 20 iterations | 11.922 s | 11.971 s |
| sd over 20 iterations | 1.308 s | 0.985 s |
| invocation means, rounds 1 to 4 | 11.40, 10.97, 12.70, 12.61 | 11.84, 12.44, 11.16, 12.45 |

Paired difference (tile minus flat) per round: +0.43, +1.47, -1.54, -0.17 s. Mean +0.05 s (+0.4% of flat), standard error
0.63 s, t = 0.08 over 4 rounds.

Reading:
- The point estimate is zero. The earlier favorable differences (-2.7%, -5% to -11%) came from run-to-run variation, which
  swings an invocation mean by 1.5 s to 2 s (13% to 17%) on one node.
- The test resolves little: with 4 rounds the standard error is 5% of the mean, so a true effect of up to about 10% in
  either direction stays compatible with the data. The result supports "no gain large enough to show at T=72" and does not
  prove "no gain".
- Decision: drop the tiling. It adds code and shows no measurable gain at one thread or at T=72.

## Memory profile of flat against tiled

VTune memory-access and hpc-performance, 72 threads, 3 iterations, single arena, 4-channel synthetic lot, both builds in
one allocation (job 40158688, `vtune-synth-pair-40158688.out`; results in `profiling/vtune-synth-{flat,tiled}-threads72-job40158688/`).

Whole program (memory-access):

| metric | flat | tiled |
|---|---:|---:|
| loads | 523.7 G | 518.1 G |
| stores | 294.3 G | 289.6 G |
| LLC miss count | 3.359 G | 3.906 G |
| DRAM bandwidth, average | 146.9 GB/s | 141.5 GB/s |
| DRAM bandwidth bound, share of elapsed | 97.7% | 98.1% |
| remote accesses | 48.7% | 47.0% |
| elapsed (hpc-performance collection) | 41.8 s | 42.8 s |

The two `band_accumulate` loops (two rows per build, 100% memory bound in both; summed over threads):

| metric | flat | tiled |
|---|---:|---:|
| CPU time (memory-access) | 227.6 s | 201.7 s |
| CPU time (hpc-performance) | 204.8 s | 176.0 s |
| loads | 17.0 G | 22.5 G |
| stores | 13.1 G | 13.1 G |
| LLC miss count | 1.58 G | 2.11 G |
| CPI (hpc-performance) | 12.1 and 11.9 | 7.3 and 5.5 |

Reading:
- The tiled loops spend 11% to 14% less CPU time, which fits the -6.0% (T=18) and -2.7% (T=72) wall-time results.
- They issue 32% more loads and 34% more LLC misses. The traffic reduction I predicted did not happen in the loop
  itself. Whole-program loads and stores are within 2% of each other, and LLC misses rise by 16%.
- DRAM bandwidth is saturated in both builds (about 98% of elapsed time bandwidth bound), so a shorter loop leaves the
  wall time nearly unchanged.
- I have no verified reason for the extra loads and LLC misses. A tile-sized chunk of each channel is read at a stride of
  `n_pixels` floats, and the prefetcher may cover that pattern worse than one long stream. This is a guess.

Caveats: one collection per build, so run-to-run variation is unknown (whole-program CPU time differs by 14% between the two
`hpc-performance` runs, 1323 s against 1505 s). I picked the loop rows by source line (flat: line 467, tiled: lines 477 and 484)
without rebuilding a line map from the exact sources that were compiled. Each build gives exactly two rows at those
lines, which matches the code structure (the bench lot subtracts the mean).

## Corrections to my earlier analysis

Claude (the pair in this step) made these statements during the step and they did not hold up:

1. Predicted tiling would cut stream traffic by 2x to 3x ("4 streams down to 2") and said to expect a bigger gain than
   -3%. Measured: no gain at one thread, and a gain only at T=18 on the 4-channel lot. The memory profile run afterwards
   shows the predicted mechanism absent: the tiled `band_accumulate` loops issue 32% more loads and 34% more LLC misses
   than the flat loops, and whole-program loads and stores match within 2%. The explanations offered before the profile
   (few channels per band, `plane` already resident in cache) were hypotheses and remain untested.
2. After the null result on the bench lot, recommended keeping the tiled version because "it should win" on production
   bands with several channels. That claim had no measurement behind it. On the synthetic lot it did not win.
3. Said production bands "probably have several channels" before reading how the slice list is built. The rule is in
   `breifast.py:675-683` (single channels, then 2, 4, 8 channel sub-bands), and the user pointed out that the number of
   slices differs between iterations. The bench lot holds one slice of one channel and represents only the single-channel
   case.
4. Said `plane` is "likely" outside L2 per thread at T=72. Never checked.
5. Compared the -3% wall-time gain with the 8.3% VTune share ("about a third of the loop"). The two figures come from
   different configurations (one thread against T=72 CPU time) and the comparison has no support.
6. Reported a 1-failure test run (67 cases) that passed in all six later runs (74 cases). The cause was never found.
   The first run may have used a stale binary.

7. Wrote "no gain" and "decision unchanged: tiling goes" after the sweep, while all eight tile sizes were 5% to 11% below
   flat. The "tiling is useless" call came from the user after job 40146936 and I repeated it without weighing the later
   data. What the data supports: no gain at one thread, and a consistent but statistically unestablished gain of about 3%
   to 9% at T=18 and T=72 on the 4-channel lot, with the loop CPU time down 14% in the profile. The evidence leans to a
   small real gain. It does not show "useless".
   The paired T=72 run then gave +0.4% (t = 0.08), so this lean was itself a reading of noise. Both the early "useless"
   and the later "probably a small gain" were drawn from comparisons too noisy to carry them. The paired design with
   alternating order and per-invocation means is what finally separated the two.

8. Estimated the lazy median at 1.5% to 3% from the VTune per-loop attribution (abs-key `select` loops at 1.6% of CPU at
   T=72, 10.3% for the signed-key loops). The one-thread measurement gave 5.4% to 6.0%. The attribution at T=72 and the
   one-thread cost disagree for this median, and I do not know why. An untested explanation: the loop takes a larger share
   when the cores are not saturating DRAM.

## Limits of the evidence

- Memory profile: one collection per build (see its caveats above).
- The tile size sweep (above) covers T=72 on the synthetic lot only. T=18, where the -6.0% appeared, was not swept.
- The synthetic lot was built by Claude: 8 planes, 4 channels (8 slices), channel k is the source image plus 10% Gaussian
  noise, group arrays repeat the source channel. It yields 0 detections, so the island labeling stage is skipped through
  the `xt::any(image_mask)` early exit. The real channel count and slice mix of the production dataset were not used.
- 5 iterations per configuration (iteration 0 dropped). At T=72 the spread is 6% to 12% of the mean, so differences of
  3% are at the edge of what the runs resolve.
- Job 40146937 lost its T=18 runs to an out-of-memory kill. The synthetic lot loads about 54 GB and each iteration copies
  it, and T=18 binds memory to one NUMA node (`numactl --membind=0`). Job 40146936 ran on a different node and finished.
  The memory use was not checked before submitting.

## Reproduce

```
# synthetic lot (about 1.5 GB on disk, about 54 GB in memory when bench loads it)
python3 make_synthetic_lot.py /home/chouh/work/src/breifast/small_test/lot_diff36_cpp /scratch/chouh/synthetic_lot_nf4 4 8

# A/B on it (run_ab.sh reads MANIFEST_DIR, ITEMS_T18, ITEMS_T72 from the environment)
MANIFEST_DIR=/scratch/chouh/synthetic_lot_nf4/ ITEMS_T18=240 ITEMS_T72=240 sbatch --export=ALL run_ab.sh 5 <dir>
python3 ../step_15/ab_summary.py ../../profiling/ab-<jobid>.out
```

## Next

- Done: the `TILED` branch, the `#define` and the `BAND_TILE` lookup are removed from `process_band`. The flat loop remains.
  The sweep and paired runs used binaries built before the removal (`profiling/ab-tilesweep/tiled`).
- Remaining step 15 candidates: #5, #6 and #7 (`compute_rms_noise_map` sqrt and column accumulators, mask copy and
  `xt::any`), then #4.
