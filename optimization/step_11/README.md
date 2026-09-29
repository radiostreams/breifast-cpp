# Step 11: histogram-allocation fix, rerun profile

Step 10 ended by tracing thread inactivity to allocation waits, naming two repeated allocations
still in the hot path: `snr` in `compute_snr_and_apply_filters` (really its three histogram
vectors) and `label_islands`. This step fixes the first and reruns step 10's profiling
methodology to check where things stand.

## The fix

`ProcessPlaneResult::histogram0/1/2` were three `std::vector<long>` on `ProcessPlaneResult`,
which is constructed fresh, locally, once per plane (`process_plane`, one call per item). Every
plane's first `compute_snr_and_apply_filters` call did `clear()` + `resize()` on empty vectors.

Fix: moved the three vectors into `ProcessBandScratch::histograms` (a `std::array<std::vector
<long>, 3>`), which is constructed once per thread and reused across every plane that thread
processes. `resize()` to the same size on every call after the first is a no-op.
`compute_snr_and_apply_filters` now takes the whole `ProcessBandScratch&` (matching
`compute_rms_noise_map`'s existing convention) instead of `snr` as a separate out-param.
`result.histograms = scratch.histograms`

## Timing

`bench <manifest> 72 5`, job 66807106 (jbm009), git HEAD `5bb01c7` plus this session's
not-yet-committed histogram and build fixes.

| iteration | compute time |
|---|---:|
| 0 (warm-up) | 12.380 s |
| 1 | 11.467 s |
| 2 | 11.297 s |
| 3 | 10.928 s |
| 4 (steady-state) | 10.384 s |

Mean of all 5 = 11.291 s. The steady-state (last) iteration, 10.384 s, sits in the same range as
step 9's post-medians-speedup baseline (10.458-10.774 s) -- consistent with removing one small
per-plane allocation, not an algorithmic change.

Single-threaded per-item (n=36): min=0.3440 s, p25=0.3619 s, median=0.3752 s, p75=0.3897 s,
max=0.5340 s, mean=0.3834 s.

## Zone breakdown

Tracy capture, 72 threads, 5 iterations, job 66807106. Self time per zone, summed across all 72
threads. Files here: `bench-threads72-iters5-job66807106.tracy` (open in the Tracy GUI),
`zones-self-time.csv` (`tracy-csvexport -e`), `zones-total-time.csv` (`tracy-csvexport`).

| zone | % self | calls | src |
|---|---:|---:|---|
| `compute_rms_noise_map` | 21.20% | 5436 | plane_lot_dispatch.cpp:121 |
| `median_of` | 19.15% | 10872 | plane_lot_dispatch.cpp:43 |
| `median_of_abs` | 18.42% | 5436 | plane_lot_dispatch.cpp:19 |
| `compute_snr_and_apply_filters` | 14.34% | 5436 | plane_lot_dispatch.cpp:236 |
| `band_accumulate` | 8.83% | 5436 | plane_lot_dispatch.cpp:411 |
| `combine` | 6.65% | 5436 | plane_lot_dispatch.cpp:434 |
| `process_band` (residual) | 5.26% | 5436 | plane_lot_dispatch.cpp:365 |
| `label_islands` | 2.78% | 453 | islands.cpp:191 |
| `mask_accumulate` | 1.57% | 5436 | plane_lot_dispatch.cpp:378 |
| `detections` | 1.15% | 453 | plane_lot_dispatch.cpp:560 |
| `pad_image` | 0.64% | 453 | plane_lot_dispatch.cpp:522 |

Medians are still the largest block (`median_of` + `median_of_abs`, 37.57% combined), unchanged
in relative share from step 10 -- this step targeted an allocation, not medians' own cost.

## Next

- `label_islands` is the other allocation site step 10 flagged, still untouched: `label_islands()`
  does `return IslandsLabeler{}(...)`, a brand-new `IslandsLabeler` (and its `parents_`/`labels_`/
  `peak_per_label_` buffers, plus a local `root_to_compact_id` vector) every call.
- Medians' own vectorization work (step 10's ranked list: `median_of_abs`'s abs-precompute, the
  second `nth_element` -> `max_element` swap, `combine`, `band_accumulate`) is still unapplied.

## VTune threading collection: measuring wait time without the collector lying

Goal: get real thread wait/idle data (Tracy has no wait-reason breakdown). `run_threading.sh`
runs `vtune -collect threading` against `bench`.

**Attempt 1: hw mode, default knobs.** Reported 63% effective CPU utilization, wait attributed
almost entirely to `[Outside any known module]`/`[Outside any task]`. But the collector itself
ran the target 50% slower than an uninstrumented Tracy run (16.9s/iter vs 11.3s) -- the wait
numbers were measuring the profiler, not the code.

**Attempt 2: sw mode.** Fails outright: this cluster blocks `ptrace` (`yama.ptrace_scope`), which
`sw` requires. `hw` is the only usable mode here.

**Attempt 3: widen sampling-interval (10ms -> 50ms), still hw.** No change in overhead (16.8s/iter).
Interval wasn't the cost.

**Attempt 4: `enable-stack-collection=false`.** Fixed it: 10.8s/iter (matches Tracy), utilization
63% -> 79.8%, reproduced at 78.4% on a 20-iteration rerun. Stack walking on every sample was the
real overhead, not sampling rate.

**Attempt 5: interval=1ms (stacks still off).** Runtime stayed native, but the collector reported
9% utilization -- implausible on its face. At 1ms the NMI sampling interrupt itself floods the
preemption accounting; the collector's *output* breaks even though the target doesn't slow down.
50-150ms are the reliable range.

**Attempt 6: interval=250/500ms.** `vtune: Error: Failed to execute sep process`, 9 of 13 attempts.
The internal log (`log/perfrun-*.log`) pins it further than the console error does:

```
runCmd: sep -dry-run -sam 250 -ec CPU_CLK_UNHALTED.THREAD:sa=2400000,...,UOPS_RETIRED.SLOTS:sample:sa=2000003 -experimental
SepDryRun RC = false
```

The interval is converted to raw PMU counter overflow thresholds (`sa=...`) and fails a dry-run
validation of that exact register programming, before any sampling starts. So it's neither "the
interval is bad" nor "this node's driver is broken" -- it's that this specific counter
configuration fails to validate on whichever node's current microcode/PMU state, which can
plausibly vary node to node even on "identical" hardware. Unresolved past this point.

**Working config:** `hw`, `enable-stack-collection=false`, interval 50-150ms.

## Reproducing

```
nix develop --command meson setup builddir-tracy -Dtracy=true
nix develop --command ninja -C builddir-tracy bench
sbatch optimization/step_11/reproduce.sh 72 5
```
