// Standalone (no Python/pybind11 involved) driver for process_band_slice_core(), for profiling
// with tools like VTune without any Python runtime, GIL, or pybind11 marshaling in the picture --
// just the actual detection algorithm, called repeatedly against realistically-sized synthetic
// data so a sampling profiler gets enough wall-clock time to attribute hotspots meaningfully.
//
// Usage: bench_process_band_slice [iterations] [image-size]
//   iterations  Number of process_band_slice_core() calls (default 30).
//   image-size  Side length of the square plane_img, e.g. 3072 to match the tron/tron-pfb
//               recipe's htc.npix default (default 3072).
//
// To profile with VTune:
//   vtune -collect hotspots -result-dir vtune-result -- ./bench_process_band_slice 50
//   vtune -report hotspots -result-dir vtune-result
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "breifast_core.hpp"

namespace
{

// Fills `arr` with float32 values drawn from N(mean, stddev), matching the rough magnitude of
// real per-pixel flux values (see the "Median noise value is 2.98e-05" style log lines from real
// pipeline runs) -- the exact distribution doesn't matter for a CPU hotspot profile, only the
// array shape/dtype does, but staying in a realistic numeric range avoids NaN/Inf edge cases that
// don't occur in practice and would otherwise waste profiled time in branches real data never hits.
xt::xarray<float> random_array(std::mt19937 &rng, const std::vector<size_t> &shape, float mean, float stddev)
{
  std::normal_distribution<float> dist(mean, stddev);
  xt::xarray<float> arr(shape);
  for (auto &v : arr)
  {
    v = dist(rng);
  }
  return arr;
}

} // namespace

int main(int argc, char **argv)
{
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 30;
  const size_t npix = argc > 2 ? static_cast<size_t>(std::atoi(argv[2])) : 3072;

  std::printf("bench_process_band_slice: %d iterations, %zux%zu plane\n", iterations, npix, npix);

  // Single-channel band (nfreq=1), matching the tron/tron-pfb recipe's typical usage -- image-htc
  // always collapses the full observing bandwidth into one channel per timeslot, so band_slices
  // in process_plane() is always just a single [0,1) band in practice (see breifast.py).
  std::mt19937 rng(42);
  xt::xarray<float> stokes_cube_template = random_array(rng, {1, npix, npix}, 0.f, 1e-4f);
  xt::xarray<float> weights({1});
  weights(0) = 1.0f;
  xt::xarray<bool> accept_mask(std::vector<size_t>{1, npix, npix});
  for (auto &v : accept_mask)
  {
    v = true;
  }
  const xt::xarray<float> channel_width = xt::full_like(weights, 763.91e6f);
  const xt::xarray<float> mean_image = random_array(rng, {1, npix, npix}, 0.f, 1e-5f);
  // selfnoise/dde_noise default to zero when unavailable in the real pipeline (see "selfnoise
  // variable not available, using zero" / "dde_noise variable not available, using zero").
  const xt::xarray<float> selfnoise_offset = xt::zeros<float>({npix, npix});
  const xt::xarray<float> dde_noise = xt::zeros<float>({npix, npix});

  // Parameters matching the values used in the manual `breifast procres` timing runs this session
  // (boxsize=400, threshold=6.0, reject-noise-threshold=10, no flux/snr cutoff images).
  const bool is_single_channel = true;
  const bool subtract_mean = true;
  const bool use_rms_noise_estimator = true;
  const int boxsize = 400;
  const float ddenoise_scaling_factor = 1.0f;
  const float full_bandwidth = 763.91e6f;
  const float reject_noise_threshold = 10.0f;
  const int histogram_detail = 0;
  const int max_sigma_bin = 200;
  const float threshold = 6.0f;

  double total_seconds = 0.0;
  double min_seconds = std::numeric_limits<double>::infinity();
  double max_seconds = 0.0;

  for (int i = 0; i < iterations; ++i)
  {
    // process_band_slice_core() mutates stokes_cube in place for single-channel bands (writes SNR
    // back), so give it a fresh copy each iteration -- otherwise later iterations would be
    // computing SNR-of-SNR instead of repeating the same representative workload.
    xt::xarray<float> stokes_cube = stokes_cube_template;

    const auto t0 = std::chrono::steady_clock::now();
    ProcessBandSliceResult result = process_band_slice_core(
        stokes_cube, weights, accept_mask, channel_width, 0, 1, is_single_channel, subtract_mean, mean_image,
        use_rms_noise_estimator, boxsize, selfnoise_offset, dde_noise, ddenoise_scaling_factor, full_bandwidth,
        std::nullopt, std::nullopt, reject_noise_threshold, histogram_detail, max_sigma_bin, threshold);
    const auto t1 = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    total_seconds += seconds;
    min_seconds = std::min(min_seconds, seconds);
    max_seconds = std::max(max_seconds, seconds);

    std::printf("  iter %2d: %.3fs (processed=%s)\n", i, seconds, result.processed ? "true" : "false");
  }

  std::printf("mean=%.3fs min=%.3fs max=%.3fs total=%.3fs over %d iterations\n", total_seconds / iterations,
             min_seconds, max_seconds, total_seconds, iterations);

  // A/B comparison of make_noise_map<float> against the production make_noise_map<double> default: how much
  // wall-clock time does the double accumulator actually cost, and how far apart are the two results, on the
  // same realistically-sized plane. See compute_integral_image()'s docstring in breifast_core.hpp for why
  // double is the production choice; this exists to make that tradeoff checkable rather than asserted.
  {
    std::printf("\nmake_noise_map<float> vs make_noise_map<double>, boxsize=%d:\n", boxsize);

    // stokes_cube_template is (1, npix, npix); make_noise_map wants a plain (H, W) plane, same as how
    // process_band_slice_core() derives plane_img for a single-channel band.
    const xt::xarray<float> plane = xt::view(stokes_cube_template, 0, xt::all(), xt::all());

    const auto t0 = std::chrono::steady_clock::now();
    const auto [noise_f, mednoise_f] = make_noise_map<float>(plane, boxsize);
    const auto t1 = std::chrono::steady_clock::now();
    const auto [noise_d, mednoise_d] = make_noise_map<double>(plane, boxsize);
    const auto t2 = std::chrono::steady_clock::now();

    const double seconds_f = std::chrono::duration<double>(t1 - t0).count();
    const double seconds_d = std::chrono::duration<double>(t2 - t1).count();

    const xt::xarray<float> abs_diff = xt::abs(noise_f - noise_d);
    const xt::xarray<float> rel_diff = xt::where(xt::not_equal(noise_d, 0.f), abs_diff / xt::abs(noise_d), 0.f);
    const float max_abs_diff = xt::amax(abs_diff)();
    const float max_rel_diff = xt::amax(rel_diff)();

    std::printf("  float:  %.3fs, mednoise=%.6g\n", seconds_f, mednoise_f);
    std::printf("  double: %.3fs, mednoise=%.6g\n", seconds_d, mednoise_d);
    std::printf("  max abs diff=%.6g, max rel diff=%.6g, double/float time ratio=%.3f\n", max_abs_diff, max_rel_diff,
               seconds_d / seconds_f);
  }

  return 0;
}
