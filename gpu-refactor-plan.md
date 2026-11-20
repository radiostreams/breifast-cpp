# GPU refactor plan

Decisions from the review of `src/gpu/` (branch `gpu-port`, 2026-11-10/11), and the order to implement them in. Goal:
less boilerplate and a clearer GPU path, without changing results except where the GPU currently diverges from the
CPU (see [Known bugs](#known-bugs-to-fix)). Nothing below is implemented yet.

## Why

The kernels are in reasonable shape; the host-side plumbing around them is not.

- Parameters are translated three times (`PlaneConfig` → `GpuPlaneJobSpec` → `GpuBandInput`, 32 fields + 12 sizes
  per band, 25 of them identical across bands) and results pass through four structs (`GpuBandResult` →
  `GpuBandHostResult` → `GpuPlaneHostResult` → `ProcessPlaneResult`). Each of the 15 per-island columns is written
  about 7 times.
- Facts like `n_pixels`, stamp size and histogram bins are stored in several structs and then cross-checked: most of
  `validate_band_geometry` and ~90 of the 100 `throw`s in `src/gpu` are validation of this kind.
- Per-island result arrays are sized to `n_pixels` (~65 B/pixel per band) though at most 4096 entries are used.
- The GPU path never checks the shapes of the user arrays it uploads, and silently returns empty results for planes
  with more than 4096 islands.
- `thrust::device_vector` temporaries and buffers allocate, fill on the legacy default stream and free with
  `cudaFree`, which serialises the per-band streams.
- Launch configuration, block reductions, atomics and radix select are duplicated; the CUDA/HIP macro layer is only
  half used.

## Target design

```
run_batch_core_gpu
  validate_plane_inputs(every plane)            shared with the CPU backend, before any work
  S worker threads, each owning one GpuSlot     S hardcoded for now (TODO: heuristic)
    GpuSlot = grow-only plane input buffers + one band workspace + host staging + Stream (declared last)
    for each plane taken from an atomic counter:
      upload inputs into the slot
      for band in reverse(config.band_slices):   same order as the CPU process_plane
        skip on host if no nonzero channel
        gpu_process_band(slot, GpuPlaneInput, GpuBand)  -> processed flag + per-band outputs
        histograms = band histograms (last processed wins), push median_noise, merge min/max sigma
        append_band_detections(...)              pure host function
      variance stats from the slot's cube, after the band loop
```

Core types (internal, `gpu_band.hpp`):

- `GpuPlaneInput`: device views and plane-wide scalars, built once per plane, shared read-only by all bands.
- `GpuBand`: only what varies per band (start, stop, selfnoise scale, beam kernel pointer, halfsize and host-computed
  sum).
- `IslandColumns<template <class> class V>`: the per-island columns declared once with a `tie()`;
  `IslandColumns<DeviceVector>` and `IslandColumns<std::vector>` for device and host.
- Workspace sizes are derived from (`GpuPlaneInput`, `GpuBand`), never stored.

## Decisions by area

### 1. Data model
- `gpu_process_band` is private; band-level behaviour is tested through the plane entry point.
- Per-island columns: templated struct (above), not an X-macro or a single buffer.
- Drop `boost_found` (computed and copied, never read). Size per-island buffers by the actual island count.
- `GpuPlaneJobSpec` goes once `DeviceVector` is thrust-free: plane setup then compiles with g++ and reads
  `PlaneConfig`/`PlaneBatchItem`/`PlaneBatchGroup` directly.

### 2. Validation
- User input (shapes, band ranges, config): one shared `validate_plane_inputs(item, group, config)` in the dispatch
  code, used by both backends; the CPU's per-band shape checks move there.
- Internal invariants (wrapper preconditions, assembly sizes): `assert`, not `throw`.
- No island cap: read the island count on the host (the sync already exists), size buffers from it, pass `n` to
  kernels by value. Removes `max_stamp_capacity`, `capacity` parameters, the `active_count` device pointer and its
  guard in every per-island kernel.
- Stamps are not materialised: per-island kernels read the plane through a device `Stamp` view (same in-plane and
  inner-box logic as `stamp_extract_kernel`). `donut_stamps`, `beam_stamps` and `gpu_stamp_extract*` go; the four
  radius-0 gathers become one `gather_at_peaks` kernel. Chunked materialised stamps are the fallback if this is slower.

### 3. Kernel launches
- `gpu_device_utils.cuh`: `block_size = 256` (also sizes every `__shared__` array), `global_index()` in `size_t`,
  `launch`, `launch_per_element`, `launch_per_island`, single-thread `launch(k, 1, 1, ...)`. Named kernels, no
  device-lambda `for_each` (keeps nsys/ncu names readable).
- The 4096-block grid-stride caps in `label_stats_scan` and `island_boost_atomic` go; they become per-element.
- Wrappers stay, one line each, as the documented and tested interface in `gpu_kernels.hpp`.

### 4. Device helpers
- `block_reduce(value, op)`: thin wrapper over `cub::BlockReduce<T, block_size>`, result broadcast through shared
  memory. `donut_std` reduces `{sum, sumsq, count}` as one struct.
- One CAS-loop `atomic_update` replaces `atomic_min_float`/`atomic_max_float`.
- `any_true` is deleted: the `nonzero_channels` check moves to the host, the pre-labeling mask check becomes
  `label_count == 0`. `gpu_scalar_reductions.cu` and its test go.
- Shared radix-select pieces (bin scan for a rank, even-count averaging) move to `radix_keys.cuh`; the donut median's
  special-cased first pass folds into its loop.
- One `combine_noise_map` kernel (nullable `rms`, device scalar base); `gpu_median_abs(global_mad)` hoisted above the
  noise-mode branch in `gpu_process_band`.
- `write_nan_kernel` goes (`n > 0` asserted). Beam kernel sum computed on the host with `xt::sum`, bit-identical to
  the CPU; `kernel_sum_kernel` goes.

### 5. Memory and scratch
- All scratch is preallocated and passed explicitly, grouped in per-algorithm structs: `MedianScratch`,
  `LabelScratch` (including `cub::DeviceScan` temp storage sized once), `IslandScratch` (grow-only).
- Workspaces are reused: exactly one workspace per active stream (the `GpuSlot` above). A slot processes a whole plane,
  bands in sequence on its stream; concurrency comes from several planes at once. `GpuBandJob`, `gpu_process_bands`,
  the thread per band and the per-band streams go.
- Number of streams is chosen by the library, not the user. For now a hardcoded constant with a TODO.
- `DeviceVector`: no thrust, plain `cudaMalloc`/`cudaFree`, uninitialised by default, `zeros()`, `from_host()`,
  `ensure(n)` (capacity ≥ n, contents unspecified after growth). Streams are created non-blocking.

### 6. Ownership
- RAII `Stream` (non-blocking create; destructor synchronises then destroys, never throws; `synchronize()` throws) and
  `DeviceVector` (moves written out so that a moved-from vector is empty) are the only classes with hand-written
  special members. Everything else is rule of zero.
- `Stream` is the last member of `GpuSlot`, so it is destroyed first and waits before buffers are freed. Comment it.

### 7. Runtime layer
- CUDA only for now; HIP stays a goal but there is no AMD hardware to test on. The `gpu*` macros are deleted;
  `DeviceVector`, `Stream` and the launch helpers are the seam for a later port. Never hardcode a warp size of 32
  (AMD wavefronts are 64); prefer CUB primitives.
- One `check(err, expr, file, line)` behind `CHECK_GPU(expr)`; `GpuError : std::runtime_error` carries the
  `cudaError_t`.
- `launch_cooperative` caches the maximum resident block count (today recomputed for every band).

### 8. Band loop and assembly
- The GPU plane loop mirrors the CPU `process_plane` (see [Target design](#target-design)). `GpuBandStatus` becomes a
  `processed` flag; `GpuBandHostResult`, `GpuPlaneHostResult`, `copy_band_jobs`, `assemble_plane_bands`,
  `assemble_plane_result` and `gpu_process_plane_result` go.
- Results are copied into pageable host staging, so each band syncs once at its result copy (syncs per band: 6 → 3).
  Pinned slot-owned staging is left for the optimisation phase.

### 9. Files and headers
- Headers 15 → 7. Public and CUDA-free: `gpu_device.hpp` (`require_cuda_device`, `GpuDeviceSummary`), `gpu_plane.hpp`
  (`run_batch_core_gpu`, `gpu_process_plane`). Internal: `gpu_runtime.hpp`, `gpu_band.hpp`, `gpu_kernels.hpp`
  (renamed from `gpu_pipeline.hpp`, conventions stated once at the top), `gpu_device_utils.cuh`, `radix_keys.cuh`.
  `GpuStream` goes; internal code uses `cudaStream_t`.
- `.cuh` = nvcc only (device code); `.hpp` = any compiler.
- Sources 15 → 9: `gpu_channel_reductions.cu`, `gpu_noise_map.cu` (windows without `compact_mask` + pointwise),
  `gpu_order_statistics.cu`, `gpu_snr_filters.cu`, `gpu_islands.cu` (+ label stats), `gpu_island_stamps.cu`,
  `gpu_process_band.cu`, `gpu_device.cpp`, `gpu_plane.cpp` (plane host + plane jobs + lot integration).
  `gpu_scalar_reductions.cu`, `gpu_band_result.cu` and `compact_mask` are deleted. The extension links
  `gpu_host_lib` instead of compiling `gpu_lot_integration.cpp`; no TBB in GPU code. Docs live only in headers.

### 10. Tests
- Tests use `DeviceVector`/`Stream` plus `gpu_test_utils.hpp` (`upload`, with a `std::vector<bool>` overload,
  `download`, `download_scalar`). No raw streams, so a failing `REQUIRE` no longer leaks one.
- Test files mirror sources: one per kernel source (nvcc), `test_gpu_plane.cpp` (plane and band behaviour, more than
  4096 islands, slot reuse), `test_gpu_detections.cpp` (host only), `test_gpu_parity.cpp`. The smoke test goes.
- New parity fixtures: a structured synthetic plane (seeded noise, Gaussian sources, non-constant noise maps, several
  channels), a band order that exercises the SNR write-back (`[wide, single0, single1]`), a many-islands case
  (~256×256, ~30k islands). Existing edge cases stay.
- Real-data `.npz` fixtures are too large to commit; the Python GPU parity test keeps depending on
  `BREIFAST_FIXTURE_DIR`.
- A second build directory with `-Db_ndebug=false` runs the suite with asserts enabled.

### Error policy (both backends)
- `validate_plane_inputs` runs on every plane before any work. During the run the first exception sets a stop flag,
  workers stop taking planes, and the exception is rethrown to Python after they finish. This replaces
  catch-print-and-return-empty in `run_batch_core` and `run_batch_core_gpu`.
- A sticky `GpuError` (e.g. illegal address) leaves the CUDA context unusable; its message says the process must be
  restarted.

### CUDA Graphs (kept possible, not adopted yet)
Decided in the optimisation phase from an nsys trace of real-size planes, against the alternative of fewer median
launches. Two constraints keep it possible:
1. Band segment B (band image → noise → medians → SNR → write-back → labeling → count readback) contains no
   allocations and no host syncs.
2. No per-plane value is passed as a kernel argument in segment B: the band-image kernel sums `weights[start..stop)`
   itself instead of taking `inv_weights_sum` (same sequential order as the host `std::accumulate`).

## Known bugs to fix

- **SNR write-back.** On the CPU, a single-channel band (when the plane has more than one channel) writes its SNR plane
  back into `stokes_cube` (`plane_lot_dispatch.cpp`, documented as mirroring breifast.py). Bands processed later and
  the spectral variance stats, computed after the band loop, see it. The GPU never writes back and computes the
  variance stats before the bands. The parity tests miss it because their cube is all ones with unit noise, so SNR
  equals the input. Fix: device-to-device copy of `workspace.snr` into the slot's cube channel right after the SNR
  stage; variance stats after the loop.
- **Island cap.** More than 4096 islands throws on the GPU; the batch runner prints the error and returns an empty
  result. The CPU has no cap. Fixed by the no-cap design in section 2.

## Implementation phases

Each phase keeps the build and the test suite green. File moves go in their own commits, separate from content
changes.

**Phase 0: safety net**
0. Benchmark baseline with `scripts/gpu_bench.py` on the current code (`scripts/kuma.sh bench ...`).
1. New parity fixtures. The structured plane runs normally; the write-back order and the many-islands case are tagged
   Catch2 `[!shouldfail]` until fixed.
2. Assert-enabled build directory (`builddir-gpu-asserts`, already created by `scripts/kuma.sh test`), noted in the
   README.

**Phase 1: mechanical, behaviour unchanged**
3. Launch helpers in `gpu_device_utils.cuh`; every wrapper converted; grid-stride caps removed.
4. CUB `block_reduce`, `atomic_update`, shared radix-select pieces, donut median first pass folded.
5. `CHECK_GPU`, `GpuError`, `launch_cooperative`, RAII `Stream`.
6. API trims: one `combine_noise_map`, beam kernel sum on the host, band-image kernel sums its own weights (graph
   constraint 2), `write_nan_kernel` and `compact_mask` deleted.
7. Kernel-side file moves only: label stats into islands, pointwise + windows into noise map, `gpu_pipeline.hpp` →
   `gpu_kernels.hpp`.

**Phase 2: memory**
8. Thrust-free `DeviceVector`.
9. `gpu_test_utils.hpp`; all tests off thrust and the `gpu*` macros; macros deleted.
10. Per-algorithm scratch structs; wrapper signatures and tests updated.

**Phase 3: validation and errors**
11. Shared `validate_plane_inputs` on every plane up front; stop-and-rethrow in both batch runners.
12. Internal checks become asserts; the ~40 precondition tests are deleted.

**Phase 4: data model and slots**
13. Structure: `GpuPlaneInput`/`GpuBand`, `IslandColumns`, `GpuSlot`, slot workers, band loop mirroring the CPU,
    `append_band_detections`, host `nonzero_channels` check. `GpuBandJob`, per-band threads, intermediate result
    structs and `GpuPlaneJobSpec` go. New host files written in their final form (`gpu_band.hpp`, `gpu_plane.cpp`).
    Stamps stay materialised in this step.
14. No cap: host island count, `n` by value, `Stamp` view, `gather_at_peaks`, `label_count == 0` instead of
    `any_true`. Remove `[!shouldfail]` from the many-islands test.
15. Write-back fix and variance stats after the loop. Remove `[!shouldfail]` from the write-back test.
16. Test files reorganised to mirror sources.

Then re-run the benchmark and compare with the Phase 0 baseline.

## Verification

The laptop has no GPU and cannot evaluate the flake (it needs the astro-nix clone on `/work/ska`). Builds, tests and
benchmarks run on kuma through [scripts/kuma.sh](scripts/kuma.sh):

```
scripts/kuma.sh test                    # build both build directories, then run all meson tests
scripts/kuma.sh test release -- gpu     # one build directory, one test
scripts/kuma.sh build                   # compile only
scripts/kuma.sh bench <manifest_dir_on_kuma> --plane-count 36
scripts/kuma.sh shell                   # interactive dev shell on a GPU node
```

Compilation must run on the `build` QOS, test runs on the `debug` QOS and benchmarks on the `normal` QOS. The script
therefore uses separate job steps: a build step without a GPU, then a step with one GPU on the `debug` QOS (tests via
`meson test --no-rebuild`, interactive shell) or on the `normal` QOS (benchmark). Never compile outside the build
step.

## Baseline (Phase 0, 2026-11-11)

`scripts/kuma.sh bench /work/ska/hchouh/src/breifast/small_test/lot_diff36_cpp --plane-count N` at commit `1de782b`
(before any Phase 1 library change), `normal` QOS, one H100, 16 CPUs, `--threads 0`, median of 3 runs after 1 warm-up.
The lot: 36 single-channel 3072 x 3072 planes, one band, boxsize 400.

| Run | Median | Detections | Peak GPU memory |
|---|---|---|---|
| GPU, 36 planes | 2.95 s | 5 | 59.3 GB |
| GPU, 144 planes | 11.62 s | 20 | 59.3 GB |
| CPU, 36 planes | 0.57 s | 5 | |

- The GPU is ~5x slower than the CPU, and 4x the planes takes 3.95x the time: no gain from concurrency.
- Memory per plane in flight is ~3.7 GB: donut stamps 4096 x 401² floats = 2.6 GB, per-island results sized to
  `n_pixels` ~0.6 GB, workspace ~0.3 GB, inputs ~0.2 GB; TBB runs up to 16 planes at once. Rerun the same three
  commands with the same CPU count after the refactor.
- A 401² stamp is 643 KB, beyond the ~227 KB of shared memory per block: the shared-memory stamp idea below does not
  apply at this boxsize.

## Deferred

- Stream-count heuristic (candidate: min(planes, slots fitting in ~80% of free memory, a small cap), failing up front
  if one slot does not fit).
- Optimisation phase: stamp loaded into shared memory once per island block, with the per-island statistics fused
  (the natural mitigation if reading stamps from the plane is slow); block-local union-find for island labeling;
  parallel and coalesced RMS row pass; fewer launches per median; CUDA Graphs; pinned host staging and inputs.
- Not adopted: CUDA Tile / cuTile (no sm_90 or C++ support as of CUDA 13.2, dense-tensor programming model,
  NVIDIA-only); `cuda::pipeline`/`memcpy_async` (no kernel streams multiple tiles through shared memory).
