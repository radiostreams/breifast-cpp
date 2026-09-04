# Step 2 — thread-scaling measurement of the step 1 migration

| | |
|---|---|
| before | `perf-step_1-before` (`1f30b6e`, "Debug + testing + bench") |
| after | `perf-step_2-after` (`568bfcc`, "Performance optimization (scratch + statically structured arrays") |
| dataset | `lot_diff36_cpp`, 3072² f32, 1080 planes |
| threads swept | 1, 2, 4, 8, 16, 32, 72 |
| iterations | 1 per thread point (single sample — see Caveats) |
| build flags | forced identical on **both** revisions: `-Dbuildtype=release -Ddebug=true` (`-O3 -g`) |
| before job | SLURM 66343646 |
| after job | SLURM 66343647 |

Unlike step 1's VTune comparison, which deliberately kept each revision's own committed
build flags to isolate the container-type change from a flags change (P1 vs P2 in
`notes/analyses/INDEX.md`), this test forces the *same* flags on both revisions. The
question here is narrower: does the step 1 migration (`xt::xarray` → `xt::xtensor<T, N>`)
actually pay off across the thread-count range the batch dispatcher runs at, not just at
72 threads?

## Results

| threads | before (s) | after (s) | speedup |
|---:|---:|---:|---:|
| 1 | 629.227 | 488.816 | 1.29× |
| 2 | 319.310 | 247.535 | 1.29× |
| 4 | 161.352 | 125.577 | 1.28× |
| 8 | 82.133 | 64.069 | 1.28× |
| 16 | 43.564 | 35.727 | 1.22× |
| 32 | 28.150 | 21.131 | 1.33× |
| **72** | **24.066** | **14.767** | **1.63×** |

Parallel efficiency, `T(1) / (n · T(n))`:

| threads | before | after |
|---:|---:|---:|
| 1 | 100 % | 100 % |
| 2 | 98.6 % | 98.7 % |
| 4 | 97.5 % | 97.3 % |
| 8 | 95.8 % | 95.4 % |
| 16 | 90.3 % | 85.5 % |
| 32 | 69.9 % | 72.3 % |
| 72 | 36.3 % | 46.0 % |

## Reading

- **1–8 threads: a flat ~1.28× speedup.** This is the raw per-thread compute win from
  static-rank indexing — no runtime stride lookups, no `svector` reload per element (see
  step 1's README for the disassembly evidence).
- **The win grows to 1.63× at 72 threads**, because the optimization also relieves the
  parallel-scaling collapse: `before`'s efficiency falls to 36 % at 72 threads while
  `after` holds 46 %. `xt::xarray`'s heap-allocated shape/strides add per-access heap
  traffic that scales worse under 72-way memory-bandwidth contention than the
  compile-time-unrolled `xt::xtensor<T, N>` offset does.
- The 16-thread row (85.5 % vs 90.3 %) is the one place `after` looks worse than `before`
  — see Caveats.

## VTune hotspots (after)

A hotspots profile of `after` (same binary as the scaling test's `after` row — see
`results/vtune-after-job66346507.provenance.txt`), 72 threads, 5 iterations, same dataset.
Full report: `results/vtune-after-job66346507.summary.txt` and `.hotspots.csv`.

| function | CPU time | % |
|---|---:|---:|
| `compute_snr_and_apply_filters` | 437.6 s | 10.9 % |
| `compute_rms_noise_map` | 411.6 s | 10.2 % |
| [Outside any known module] | 377.5 s | 9.4 % |
| `std::nth_element` (via `xt::median`'s partition) | 254.5 s | 6.3 % |
| `xt::median<...>` | 235.0 s | 5.8 % |

`nth_element` + `xt::median` together are 12.2 % of CPU time — this is step 1's README's
"P3" item (`xt::median` → `std::nth_element` on `.data()`), predicted at 19.2 % on the
pre-migration baseline and now down but still the largest single opportunity after the two
compute kernels at the top.

This was the third VTune attempt on this dataset before getting a usable capture:

1. Job 66343391 — the `bench` binary was rebuilt (switching build flags mid-conversation)
   while this job was actively collecting, inside its own collection window. Post-hoc
   symbol resolution against the now-different on-disk binary produced nonsense: e.g.
   `nlohmann::json` parsing (which only ever runs once, single-threaded, at startup)
   attributed 291 s of CPU time in a run whose `Paused Time` accounting shows that phase
   was never even sampled — arithmetically impossible under either bound. Not versioned.
2. Job 66346501 — VTune's own trace file was corrupted (`Cannot load data file ... Data
   file is corrupted`), 47 s elapsed, 0.4 s of CPU time captured, `bench` never printed
   its startup line. Not versioned.
3. Job 66346507 — clean: `Paused Time` (89.6 s) matches elapsed minus `bench`'s own
   reported total (155.2 − 66.2 ≈ 89.0 s), and the top hotspots are real compute kernels,
   not parsing code. This is the one kept in `results/`.

Step 1's baseline VTune runs (66340861, 66343033) had their own, different reliability
problems — see step 1's README "Known issues". Treat any VTune result from this cluster as
unverified until its `Paused Time`/elapsed cross-check and `Result Size` have been sanity
checked against the program's own log.

## Caveats

- **Single sample per point** (`iterations=1`). The 16-thread dip above is plausibly
  measurement noise rather than a real regression; re-run with `iterations=3-5` before
  citing these numbers in a write-up.

## Reproducing

Both revisions are now real tags, built the same way via `scaling_rev.sh` (worktree +
`build-fix.patch`, forced `-Dbuildtype=release -Ddebug=true`):

```bash
./scaling_rev.sh perf-step_1-before before 1 "1 2 4 8 16 32 72"
./scaling_rev.sh perf-step_2-after after 1 "1 2 4 8 16 32 72"
```

`perf-step_2-after` (`568bfcc`) predates the build-config-fix commit
(`a303c96`, "Fix meson/nix configuration so the project actually builds"), so it still
needs `build-fix.patch` to configure — `scaling_rev.sh` only skips the patch when the
checked-out `meson.build` already defines `itt_dep`.

Each job writes `profiling/scaling-<label>-job<id>.csv` plus a matching
`.provenance.txt` recording the exact binary mtime, dataset, and thread list. Copies of
both runs' outputs are kept in `results/` since `profiling/` is gitignored.
