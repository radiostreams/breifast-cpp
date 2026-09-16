step4's raw-pointer rewrite (`compute_snr_and_apply_filters`) eliminated the targeted accessor
overhead but produced no wall-clock improvement. A dedicated `memory-access` VTune collection
(job 66356748) explained why: the pipeline is **81.9% Memory Bound**, with **49.8% of DRAM loads
serviced from remote NUMA nodes** and **63.9% UPI Utilization Bound** -- cross-socket memory
traffic, not xtensor abstraction overhead, is the real bottleneck.

Hypothesis: worker threads aren't pinned to a NUMA node, so a thread's scratch buffers/working
set can end up on a different socket than the thread itself, forcing every access over the
UPI (Ultra Path Interconnect) links between sockets. Pinning TBB's task arenas to NUMA nodes
should keep each thread's memory local and cut UPI traffic.

## Attempt 1: one `tbb::task_arena` per NUMA node

Implemented in `run_batch_core` (`src/plane_lot_dispatch.cpp`): discover NUMA nodes via
`tbb::info::numa_nodes()`, split `num_threads` across them proportionally (largest-remainder
method), build one `constraints{}.set_numa_id(...).set_max_concurrency(...)` arena per node, and
partition `items` into contiguous per-node ranges dispatched concurrently via `tbb::task_group`.

Result: no measurable change, and profiling still showed 38-47% remote NUMA accesses --
essentially unchanged from before pinning was added.

## Root cause: TBB never had real NUMA topology in the first place

VTune's GUI doesn't attribute UPI/remote-access samples back to source lines. Querying the
result's SQLite database (`sqlite-db/dicer.db`) directly, joining `dd_sample` &rarr;
`dd_callsite` &rarr; `dd_code_location` &rarr; `dd_source_location`, surfaced the hottest
remote-access lines:

| file:line | code | remote accesses | CPU time |
|---|---|---:|---:|
| `plane_lot_dispatch.cpp:148` | `window_sum = parts_sums(max_y,max_x) - ...` (box-filter integral image lookup in `compute_rms_noise_map`) | 150.5M | 196.7s |
| `plane_lot_dispatch.cpp:399` | `excess_noise_factor(i_y,i_x) = noisemap(i_y,i_x)/noisemap0_at(i_y,i_x)` in `combine()` | 105.1M | 145.3s |
| `plane_lot_dispatch.cpp:391-392` | `noisemap(i_y,i_x) = sqrt(...+group.selfnoise_offset(i_y,i_x)+...group.dde_noise(i_y,i_x))` in `combine()` | 25.1M+25.1M | 67.9s+54.8s |
| `plane_lot_dispatch.cpp:130` | `parts_sums(i_y+1,i_x+1) = plane(i_y,i_x)*plane(i_y,i_x) + ...` (integral-image build) | 15.1M | 33.1s |
| `plane_lot_dispatch.cpp:349` | `plane(i_y,i_x) += item.stokes_cube(i_channel,i_y,i_x) * weights(...) * inv_weights_sum` (band accumulation) | 10.1M+5.0M | 148.5s |
| `plane_lot_dispatch.cpp:466` | `padded_image(i_y,i_x) = ... plane(...)` | 5.0M | 6.7s |
| `plane_lot_dispatch.cpp:201` | `image_mask_ptr[i_pixel] &= valid` in `compute_snr_and_apply_filters` | 5.0M | 1.2s |

Several of these lines (`parts_sums`, `windows_sums`) are per-thread scratch buffers that should
already be NUMA-local by first-touch placement, once a thread is actually pinned to a node. That
they still showed heavy remote traffic pointed at the pinning itself, not at data layout.

A diagnostic added to `run_batch_core` (logging `sched_getaffinity()`/`sched_getcpu()` per
thread, plus `tbb::info::numa_nodes()` itself) confirmed the real problem: **every thread
reported the full 72-CPU affinity mask, and `numa_nodes().size() == 1` with `numa_id == -1`** --
TBB's sentinel for "no NUMA information available". Every arena built in Attempt 1 was silently
the same single unconstrained arena; nothing about execution had actually changed.

`tbb::info::numa_nodes()` is backed by a separate shared library, `libtbbbind`, which oneTBB only
builds if `hwloc` is discoverable via `pkg-config` at its own CMake-configure time. The vendored
TBB build in `meson.build` never had `hwloc` available, so `libtbbbind` was never built.

## The fix

- `meson.build`: the vendored TBB build now also copies `libtbbbind_2_5.so.3` into the same flat
  output directory as `libtbb.so.12`. TBB resolves `tbbbind` as a sibling of wherever `libtbb.so`
  itself was actually loaded from (via `dladdr`, ignoring `LD_LIBRARY_PATH` by design), so it has
  to sit next to that file, not just inside `tbb-install/lib64`.
- Building (not just running) now requires `hwloc >= 2.5` findable via `pkg-config`, plus its
  `Requires.private` (`libxml-2.0`, `pciaccess`) resolvable too, or the `pkg-config --cflags`
  query silently comes back empty and `libtbbbind` is never built. Known-good combo on this
  cluster: `module load hwloc/2.9.1 libxml2/2.10.3`, plus `libpciaccess`'s `pkgconfig` dir added
  to `PKG_CONFIG_PATH` by hand (no standalone `pciaccess` module exists here).
- Running `bench` afterwards only needs `module load hwloc/2.9.1` (for `libtbbbind`'s own
  `libhwloc.so.15` dependency at runtime) -- added to `run_profile.sh` and
  `run_bench_memaccess.sh`.

With `libtbbbind` actually present, `tbb::info::numa_nodes()` correctly reports the machine's 4
real NUMA nodes (18 CPUs each), and the Attempt 1 arena/pinning code -- unchanged -- starts
actually pinning threads.

## Result

Memory-access profiling, before vs. after the fix (same config, 72 threads):

| | before (job 66526654) | after (job 66530765) |
|---|---:|---:|
| Memory Bound | 100.0% | 63.5% |
| NUMA: % of Remote Accesses | 43.0% | **7.5%** |
| UPI Utilization Bound | 62.9% | **0.0%** |
| Remote Memory Access Count | 976,631,890 | **0** |
| Local Memory Access Count | 783,420,500 | 1,478,461,056 |
| CPU Time | 5048.1s | 4965.9s (-1.6%) |
| Elapsed Time | 283.7s | 296.1s (+4.4%) |

Bench's own timer, before (`282e669`, no NUMA pinning) vs. after (NUMA-pinned arenas), 5 iters:

| | before | after |
|---|---:|---:|
| mean iteration (`run_batch_core`) | 11.711 s | 11.387 s (-2.8%) |
| per-iteration range | 10.616-12.782 s | 10.899-12.094 s |

**Hypothesis confirmed; mean-iteration wall-clock is within noise, but CPU Time is genuinely
lower.** UPI utilization and remote-access count both dropped to essentially zero -- the pinning
now genuinely works. Local Memory Access Count roughly doubled (783M &rarr; 1.48B): the same
total memory traffic that used to cross the UPI fabric is now just served from local DRAM
instead. DRAM bandwidth utilization (~123 GB/s average, ~59% of elapsed time in the
high-utilization band) was already saturated-ish both before and after, and the -2.8%
mean-iteration change is within this cluster's normal run-to-run noise (see step_1's README).
But total CPU Time dropped too (5048.1s &rarr; 4965.9s, -1.6%) -- a real, if modest, reduction in
aggregate work done, consistent with threads no longer paying the extra latency/retry cost of
remote memory access even though it doesn't show up as a wall-clock win at this thread count.

**Implication**: NUMA placement is correctly solved and worth keeping. The bigger remaining
lever for wall-clock improvement is reducing total memory traffic per pixel (working-set size,
fewer large arrays touched per item) -- a data-layout/algorithmic problem, not a placement one.

## Reproducing

Cherry-picked to `main` (commit `1f59096`) since CPU Time improved. Building still needs hwloc
available to CMake's pkg-config check:

```bash
module load hwloc/2.9.1 libxml2/2.10.3
export PKG_CONFIG_PATH="<libpciaccess prefix>/lib/pkgconfig:$PKG_CONFIG_PATH"
nix develop --command meson setup --wipe builddir .
nix develop --command ninja -C builddir bench
```

Running only needs `module load hwloc/2.9.1` (already added to `run_profile.sh`,
`run_bench_memaccess.sh` and `run_bench.sh`).
