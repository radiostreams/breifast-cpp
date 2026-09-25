# Step 9: two-pass separable rewrite of `compute_rms_noise_map` — negative result

## Hypothesis

Replace the summed-area table (`main`, 75.6 MB `parts_sums` scratch/thread) with a two-pass
separable box filter (horizontal running sum -> `row_window_sums`, then vertical running sum),
column-blocked to keep running sums L1-resident. Predicted traffic: 188 MB vs 302 MB (1.6x).

## Result

| | steady-state, 72 threads | single-thread per-item |
|---|---:|---:|
| main (SAT) | 10.764 s | 0.3623 s |
| two-pass, no blocking | 10.458 s | 0.3788 s |
| two-pass, blocked (512) | 10.774 s | 0.3883 s |

Two-pass with no blocking: **2.8% faster** at 72 threads, **4.5% slower** single-threaded.
Column blocking cost 3.6% and is removed.

## Why the prediction was wrong

Blocking was sized around the window footprint (400x400x4B = 625 KB), assumed too big for
L1/L2. Never checked the actual resident structure: `column_accumulators` at full plane width
(3072 x 8B = 24.6 KB) fits inside Icelake's 48 KB L1d on its own. The capacity constraint the
whole design was built around didn't exist. Blocking only added a strided access pattern to
`row_window_sums` and lost cache-line utilization.

Block-size sweep (64/128/256/512/1024/2048/3072), 72 threads, steady-state:

| block_size | steady mean |
|---|---:|
| 64 | 10.928 s |
| 512 | 10.774 s |
| 1024 | 10.520 s |
| 3072 (no blocking) | 10.381 s |

Monotone toward "no blocking at all."

The 2.8% net gain is real but far short of the 1.6x traffic reduction predicted. This function
is not memory-bandwidth-bound the way the LLC-miss profiling suggested; the ~7% spread across
all four implementations measured so far (main, BOX_VERSION, strip ring, two-pass) says this
computation is close to its floor regardless of algorithm.

## Instrument mismatch

Single-thread per-item time (used throughout steps 6-8 because it reproduces to ~0.2%) gave the
wrong sign here: ranked two-pass 4.5% slower while it was 2.8% faster at 72 threads. It isolates
compute, not memory contention, so it's the wrong metric for changes targeting bandwidth. Any
future memory-bound change needs the 72-thread steady-state number, not this one.

## Kept

- 2.8% faster at 72 threads
- 37.9 MB less scratch/thread (2.7 GB across 72 threads)
- shorter warm-up (main: ~4 iterations to steady state, two-pass: ~2)
- simpler code than BOX_VERSION's ring buffer

## Not pursued further

`compute_rms_noise_map` is not where the time is: medians are 34.8% of per-item time (step 8,
Tracy), and the two bit-exact deletions identified in step 8 (`excess_noise_factor` array,
median floor pass) are worth ~8%, both larger than anything this rewrite produced.
