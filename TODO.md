# TODO

## 1. ~~Unit tests for the individual C++ functions~~ (done)
`tests/cpp/test_breifast_core.cpp` now covers `compute_integral_image`,
`box_sum`, `make_noise_map` (both `Accum` variants), `histogram_counts`,
`estimate_noise_and_snr`, and `apply_rejection_cutoffs` in isolation, using
Catch2 -- a standalone C++-native test executable (same pattern as
`bench.cpp`, no pybind11 needed) rather than exposing them
to Python. Catch2 is built from source by meson (`catch2_build` in
meson.build), not pulled from nixpkgs -- the prebuilt nixpkgs binary needs a
newer glibc than the system one this project links against (see the comment
above `catch2_build`), the same reason TBB is also built from source.

## 2. Unit benchmarks (per-function, not just end-to-end)
`bench.cpp` currently only times the full
`process_band_slice_core()` call, plus the one-off `make_noise_map<float>` vs
`<double>` A/B added today. Extend with per-function microbenchmarks
(`compute_integral_image`, `box_sum`, `histogram_counts`, etc.) to find which
piece actually dominates at various `boxsize`/image-size combinations.

## 3. Xtensor performance bottlenecks
Today's VTune profile found `xt::median(xt::abs(plane_img))` passing a lazy
expression straight into `xt::median()` (walked via the slow generic stepper,
~99% of profiled time) -- fixed. Re-profile the *current* code with VTune to
see what's the new dominant hotspot now that that's gone; check for other
spots where a lazy xtensor expression gets iterated without first being
assigned into a concrete `xt::xarray` (same class of bug).

## 4. Manual (non-xtensor) implementation, for comparison
Explored conceptually today but not started. Write a hand-rolled version of
`process_band_slice_core()` (raw loops/pointers, no xtensor) to A/B against
the current xtensor version for both performance and code complexity. Keep
the existing xtensor version as the correctness reference -- both should be
checked against `tests/test_process_band_slice_fixtures.py`'s fixtures.

## 5. Extend the C++ port to more of the pipeline
Only `process_band_slice` (the per-band-slice body of `process_plane`) is
ported so far. Island detection and per-island statistics (`scipy.ndimage.label`
/ `find_objects` and everything downstream in `breifast.py`'s `process_plane`)
are still pure Python -- `xtensor` has no connected-component labeling
primitive, so that would need either a hand-rolled union-find/flood-fill or a
different library. Scope this out before starting: how much of the remaining
Python runtime is actually spent there vs. elsewhere, worth profiling the
Python side first.

---
*Context for picking this back up:* `breifast_core.hpp` is the
Python/pybind11-independent core (shared by `breifast_helpers.cpp`'s pybind11
wrapper and `bench.cpp`'s standalone VTune driver). Real
regression fixtures live on Nextcloud, not in git -- `./scripts/fetch_fixtures.sh`
to pull them down (see `README.md`).

## Checklist

- [x] Unit tests for the individual C++ functions
- [x] Unit benchmarks (per-function, not just end-to-end)
- [x] Xtensor performance bottlenecks
- [x] Manual (non-xtensor) implementation, for comparison
- [x] Extend the C++ port to more of the pipeline
