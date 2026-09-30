# Step 13: remaining full-array passes per band

Step 12 showed the code is limited by DRAM bandwidth per NUMA domain, and radix medians cut 25% at
72 threads by removing traffic. This lists the full-array passes left in `process_band`, largest
avoidable traffic first. One 3072 x 3072 float plane is 37 MB.

## Headline: where the code stands after #1, #2, #3 and block removal

| comparison | before | now | speedup | source |
|---|---:|---:|---:|---|
| Python, bench lot, 72 threads, same node | 102.35 s | 6.245 s | 16.4x | `python-vs-cpp-66904787.out` |
| first C++ (`1f30b6e`), bench, 72 threads | 24.888 s | 6.379 s | 3.9x | step 8, `ab-66899190.out` |
| Python, pipeline on `test.ms`, 37 threads | 41.98 s | 9.76 s | 4.3x | `pipeline-*-6690394[45].out` |

The 72-thread rows are the at-scale figures. `test.ms` is a reduced dataset: 37 intervals, 148 items per scale,
and the pipeline caps both paths at 37 threads (`ncpu = min(cpu_count, nt)`), so that row is an end-to-end check.
Details in the last two sections.

## Full-array passes left

| # | pass | traffic now | fix | saves | effort |
|---|---|---|---|---|---|
| 1 | `snr` written back into `item.stokes_cube` when `band.size() == 1` | read + write, 74 MB | skip it when nothing reads the cube afterwards | 74 MB | small, needs a semantics check |
| 2 | `windows_sums` clipped to its own median at the end of `compute_rms_noise_map` | read + write, 74 MB | apply `max(ws, median)` where `combine` reads it | 74 MB | small |
| 3 | `band_accumulate`: `plane.fill(0)`, then `+=` per channel | 3 planes for one channel | assign on the first channel, accumulate on the rest | 74 MB | small |
| 4 | pass 1 of each radix median | 1 read x 3 medians | build the histogram in the loop that produces the array | 111 MB | medium, adds histogram increments to those loops |
| 5 | boost, per island | labels 75 MB + `baseline_flux` 37 MB | per-island pixel lists from `label_islands` | ~112 MB per island (~150 islands per 1080 items) | medium |

## Notes

- **#1:** the binding copies the cube into an `xtensor`, and the reversed band order writes each
  single-channel band back after it has been consumed, so the write-back looks dead in C++.
  `compute_spectral_variance_stats` reads `item.stokes_cube` for multi-channel items, and
  `breifast.py` overwrites in place the same way, so matching Python probably means keeping the
  write only when variance stats run. Check against `breifast.py` before removing it.
- **Block size:** `compute_rms_noise_map` still defaulted to `block_size = 512`. Step 9 measured no
  blocking (3072) as fastest and recorded blocking as removed. Only the step 9 sweep scripts
  override `BREIFAST_BLOCK_SIZE`, so every run since used 512. Removed, see below.

## Order

1. #2 and #3: local changes, no semantic questions, about 148 MB per band together.
2. #1, once the semantics are settled.
3. #4, then #5.

## #1 done: write-back gated on multi-channel (`counters-66853138.out`)

`breifast.py` computes the spectral variance stats from `stokes_cube` in SNR units ("the slice has
been normalized to units of SNR at this point"), so the write-back is required whenever those stats
run (`nfreq > 1`). They are its only reader in C++: the binding copies the cube, and the reversed
band order consumes each channel before it is overwritten. The write-back is now gated on
`item.nonzero_channels.size() > 1`, the same condition as the variance stats. Every benchmark item is
single-channel.

Test: `test_process_plane.cpp` pins multi-channel variance stats against numpy on the SNR cube.
Removing the write-back fails that test alone, and nothing else.

Mean of 2 iterations, vs step 12's radix run (`counters-66850675.out`): T=1 -2%, T=18 bound -7%,
T=72 single arena 7.84 -> 6.74 s (-14%), T=72 4 NUMA arenas -9%.

## #2 done: median floor applied inside `combine` (`ab-66854248.out`)

`compute_rms_noise_map` returns the box-filter RMS only. `process_band` takes its median and passes
`combine` a per-pixel accessor, `max(rms[i], floor)` for RMS and `noise_global_mad` for global MAD.
The accessor inlines to one `maxss` in the loop. `excess_noise_factor` reads the map at each island
peak and applies the same floor.

Tests: `test_compute_rms_noise_map*.cpp` apply the floor through a local helper, so their Python
references stay valid. Dropping the floor from the accessor fails the three Python-reference
integration tests.

A cross-node run (`counters-66854183.out`) showed +5% at 72 threads and -4 to -5% on one domain,
within the node-to-node spread. The same-node A/B settles it:

| config | before | after | change |
|---|---:|---:|---:|
| T=18, domain 0, local | 5.531 +- 0.069 s | 5.172 +- 0.065 s | -6.5% |
| T=72, single arena | 6.799 +- 0.084 s | 6.558 +- 0.075 s | -3.5% |

## #3 done: `band_accumulate` assigns on the first channel (`ab-66855315.out`)

The first channel assigns `cube * w / sum(w)` (minus the mean term), the rest accumulate. `plane.fill(0)`
is gone.

| config | before | after | change |
|---|---:|---:|---:|
| T=18, domain 0, local | 5.174 +- 0.090 s | 5.112 +- 0.068 s | -1.2% |
| T=72, single arena | 6.465 +- 0.146 s | 6.289 +- 0.080 s | -2.7% |

## Block size: blocking removed from `compute_rms_noise_map` (`ab-66855316.out`, `ab-66899190.out`)

The vertical pass keeps one row of column accumulators over the full width. Setting
`BREIFAST_BLOCK_SIZE=3072` on the #3 build (`ab-66855316.out`) and the unblocked build against the #3
build at 512 (`ab-66899190.out`) agree:

| config | env 3072 vs 512 | unblocked vs 512 |
|---|---:|---:|
| T=18, domain 0, local | -3.5% | -3.0% |
| T=72, single arena | -2.7% | -3.4% |

Unblocked build: 4.975 +- 0.088 s at T=18, 6.379 +- 0.097 s at T=72.

## Measuring small changes: same-node A/B (`run_ab.sh`)

Different jobs land on different nodes, with about 5% spread in wall time, as large as the effects
now being measured. `run_ab.sh [iterations] [ab_dir]` runs `<ab_dir>/before/bench` and
`<ab_dir>/after/bench` in one job, interleaved before/after/after/before/before/after, 5 iterations
each, dropping iteration 0. Per-run means agree within about 1%, which resolves 1-2% effects. The
"before" binary is built in a detached worktree (`../breifast-cpp-worktrees/ab-before`, with `tracy/`
symlinked in) and copied without `libtbbbind`, so both run a single arena.

## Pipeline on the reduced `test.ms`: Python vs C++ (`pipeline-python-66903944.out`, `pipeline-cpp-66903945.out`)

Step 8's recipe (`run-python-big.sh` / `run-cpp-big.sh` in `breifast/small_test`: `tron-pfb`, `test.ms` +
`deep-image.fits`, `stokes: IQUV`, 3 scales), run from `/scratch/chouh/breifast_small_test` on bigmem nodes
(jbm031, jbm032), `--exclusive`. The C++ side is a cp313 wheel of this step's code, built in
`../breifast-cpp-worktrees/py-compare` with the step 6 `flake.lock` (the current lock gives Python 3.14), with
an empty `libtbbbind` so it runs a single arena. Running from `/work` stalled on GPFS metadata (about 2 zarr
files/s, 80 min for a 17 min pipeline).

Python time is the sum of the `python_plane` lot wall times, C++ the sum of the `cpp_plane` batch compute times,
as in step 8.

| | step 8 | now |
|---|---:|---:|
| Python | 43.41 s | 41.98 s |
| C++ | 12.82 s | 9.76 s |
| speedup | 3.4x | 4.3x |

Both find the same 5 detections. The SNR columns differ from Python by up to 3.2e-4 relative, as in step 8. C++
matches step 8's C++ to 1.1e-7.

`breifast.py` sets `ncpu = min(cpu_count, nt)`, and `test.ms` has 37 intervals, so both run 37 threads: Python a
38-worker pool over lots of 72 items, C++ one `process_lot` of 148 items per scale.

## Bench lot at 72 threads: Python vs C++ (`python-vs-cpp-66904787.out`)

`scripts/python_bench.py` runs `breifast.py`'s own `process_plane`, extracted from its source at runtime, on the
bench lot (`lot_diff36.pkl`, 36 planes x 30 replicas, pipeline dtypes), all 1080 items on one 72-worker pool.
`run_python_vs_cpp.sh` runs bench, Python, bench in one job (jst113), data on `/scratch`. Both find 150
detections.

| | mean | per item |
|---|---:|---:|
| Python, 72 threads (3 iterations) | 102.35 +- 0.74 s | 94.8 ms |
| C++ bench, 72 threads, single arena (8 iterations) | 6.245 +- 0.135 s | 5.8 ms |
| speedup | 16.4x | |

Python's per-item time here matches the 37-thread pipeline (96.3 ms), so it gains nothing from the extra threads.
The pipeline's C++ time (22.0 ms per item) includes `process_lot`'s array conversion and result marshaling and
runs 148-item batches on 37 threads.
