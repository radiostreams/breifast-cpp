The last step gave nothing (memory accesses)
Now we are looking into improving compute performance nonetheless through SIMD.

We start by identifying the most compute intensive loops that could be vectorized.

Claude starts by identifying a few candidates based on static code analysis. We correct it and asks it to run an actual profiling to do evdience-based identification.
After running a tracy analysis, it realizes that the candidates it identified were not hotspots.

## Profiling result

Tracy capture, 72 threads, 3 iterations, node jbm015, job 66737513. Files here:
`bench-threads72-iters3-job66737513.tracy` (open in the Tracy GUI), `zones-self-time.csv`
(`tracy-csvexport -e`), `zones-total-time.csv` (`tracy-csvexport`).

Reproduce with `sbatch optimization/step_10/reproduce.sh` at tag `perf-step_10-profile`. Its
header lists the two prerequisites: a Tracy 0.14.2 (81c50055) checkout at `tracy/` with capture
and csvexport built, and `ninja -C builddir-tracy bench` after `meson setup builddir-tracy
-Dtracy=true`.

Self time per zone, summed across all 72 threads:

| zone | % self | calls | src |
|---|---:|---:|---|
| `compute_rms_noise_map` | 18.87% | 3276 | plane_lot_dispatch.cpp:112 |
| `median_of` | 18.48% | 6552 | plane_lot_dispatch.cpp:38 |
| `median_of_abs` | 17.38% | 3276 | plane_lot_dispatch.cpp:19 |
| `compute_snr_and_apply_filters` | 16.63% | 3276 | plane_lot_dispatch.cpp:228 |
| `combine` | 9.96% | 3276 | plane_lot_dispatch.cpp:426 |
| `band_accumulate` | 8.00% | 3276 | plane_lot_dispatch.cpp:403 |
| `process_band` (residual) | 5.19% | 3276 | plane_lot_dispatch.cpp:357 |
| `label_islands` | 2.50% | 273 | islands.cpp:191 |
| `mask_accumulate` | 1.45% | 3276 | plane_lot_dispatch.cpp:370 |
| `detections` | 0.97% | 273 | plane_lot_dispatch.cpp:562 |
| `pad_image` | 0.56% | 273 | plane_lot_dispatch.cpp:524 |

## What the static analysis got wrong

Top candidate was `compute_spectral_variance_stats`, ranked first on trip count
(`n_channels * n_y * n_x`). It has a `ZoneScoped` and does not appear in the trace at all: it is
gated on `item.nonzero_channels.size() > 1` (plane_lot_dispatch.cpp:673), never true for this
dataset. Ranked a loop that executes zero times.

`compute_snr_and_apply_filters` was demoted as "not worth pursuing first" for being branch-heavy.
It is 16.63%, the largest single dense loop in the file.

## Measured candidates

Medians are **35.86%** combined, the largest block, and contain two vectorization problems:

1. `median_of_abs` (17.38%) passes `cmp = [](T a, T b) { return std::abs(a) < std::abs(b); }` to
   `nth_element`. At n = 9437184 and ~2-3n comparisons, that is 40-50M scalar `fabs` calls inside
   a comparator that cannot vectorize. Taking absolute values once into `scratch` as a separate
   elementwise pass, then selecting with plain `<`, replaces them with one vectorizable pass.

2. Both medians run a second `nth_element` for the even-n case (lines 31 and 50). The range is
   `[begin, begin + mid)`, exactly `mid` elements, requesting the last position: the maximum of
   the lower half. `n` is always even here, so this always executes, running full introselect
   partitioning over 4.7M elements where `std::max_element` is one vectorizable linear scan.

Ranked:

| # | target | % self | note |
|---|---|---:|---|
| 1 | `median_of_abs` abs-precompute | 17.38% | removes 40-50M scalar `fabs` from comparator |
| 2 | 2nd `nth_element` -> `max_element` | 35.86% | small change, touches both medians |
| 3 | `combine` | 9.96% | two branchless elementwise loops; `excess_noise_factor` half already slated for deletion (step 8) |
| 4 | `band_accumulate` | 8.00% | elementwise multiply-add, hoistable loop-invariant branch |
| 5 | `compute_snr_and_apply_filters` | 16.63% | biggest dense loop, hardest (histogram scatter) |

## Vectorization work
After further analysis, Claude pointed out a few vectorization sites and suggested adapting them and compile options to allow GCC to auto-vectorize.
We suggested using xsimd where relevant instead of only relying on GCC (results can vary from a version to another).

Claude also suggesting rolling the intermediate array excess_noise_factor to compute it on the fly only where needed.
It made refactoring suggestions to remove the portions of code involved but also made a useless suggestion to keep a pointer to a function result that wasn't needed for no good reason.