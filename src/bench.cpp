// Standalone (no Python/pybind11 involved) driver for the two core algorithms in
// breifast_core.hpp -- process_band_slice_core() and process_plane_core()/process_plane_batch_core()
// -- for profiling with tools like VTune without any Python runtime, GIL, or pybind11 marshaling in
// the picture, against realistically-sized synthetic data.
//
// The "plane" mode calls process_plane_batch_core() directly for its "batch"/parallel measurement
// (the exact same function process_plane_batch()'s pybind11 wrapper in breifast_helpers.cpp
// delegates to), rather than reimplementing a separate parallel loop here -- this benchmark exists
// to exercise the real dispatch code, not a stand-in for it. Only the "sequential" baseline in that
// mode is this file's own loop, since it isn't standing in for any real parallel-dispatch code path
// -- it's just process_plane_core() called directly, with no batching involved at all.
//
// Usage: bench <band|plane> [args...]
//   bench band [iterations] [image-size]
//     iterations  Number of process_band_slice_core() calls (default 30).
//     image-size  Side length of the square plane_img, e.g. 3072 to match the tron/tron-pfb
//                 recipe's htc.npix default (default 3072).
//   bench plane [iterations] [image-size] [items-per-batch] [num-threads]
//     iterations       Number of (sequential-loop, process_plane_batch_core) comparisons (default 5).
//     image-size       Side length of the square plane, default 3072.
//     items-per-batch  Number of synthetic (interval, stokes) items in one batch, i.e. one
//                      process_plane_batch()-style call (default 8).
//     num-threads      Caps the TBB thread pool via tbb::global_control, forwarded to
//                      process_plane_batch_core() exactly as process_plane_batch() would (default 0
//                      = TBB's own default, all hardware threads).
//
// To profile with VTune:
//   vtune -collect hotspots -result-dir vtune-result -- ./bench band 50
//   vtune -collect hotspots -result-dir vtune-result -- ./bench plane 5 3072 8
//   vtune -report hotspots -result-dir vtune-result
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "breifast_core.hpp"

namespace
{

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

int run_band_slice_bench(int argc, char **argv)
{
  const int iterations = argc > 0 ? std::atoi(argv[0]) : 30;
  const size_t npix = argc > 1 ? static_cast<size_t>(std::atoi(argv[1])) : 3072;

  std::printf("bench band: %d iterations, %zux%zu plane\n", iterations, npix, npix);

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

  // A/B comparison of process_band_slice_core() (xtensor) against process_band_slice_core_serial() (hand-rolled
  // raw loops/pointers, no xtensor) -- see TODO.md item 4. Each variant gets its own fresh copy of
  // stokes_cube_template per iteration (both mutate their cube in place), timed via ScopedTimer so
  // results accumulate across all `iterations` calls and print as one aggregate report at the end,
  // rather than a line per call.
  {
    std::printf("\nprocess_band_slice_core (xtensor) vs process_band_slice_core_serial (manual), boxsize=%d:\n",
               boxsize);
    ScopedTimer::reset();
    float max_snr_diff = 0.f;
    float max_noisemap_diff = 0.f;
    int processed_count = 0;

    for (int i = 0; i < iterations; ++i)
    {
      xt::xarray<float> cube_xtensor = stokes_cube_template;
      xt::xarray<float> cube_manual = stokes_cube_template;

      ProcessBandSliceResult result_xtensor;
      {
        ScopedTimer timer("xtensor");
        result_xtensor = process_band_slice_core(
            cube_xtensor, weights, accept_mask, channel_width, 0, 1, is_single_channel, subtract_mean, mean_image,
            use_rms_noise_estimator, boxsize, selfnoise_offset, dde_noise, ddenoise_scaling_factor, full_bandwidth,
            std::nullopt, std::nullopt, reject_noise_threshold, histogram_detail, max_sigma_bin, threshold);
      }

      ProcessBandSliceResult result_manual;
      {
        ScopedTimer timer("manual");
        result_manual = process_band_slice_core_serial(
            cube_manual, weights, accept_mask, channel_width, 0, 1, is_single_channel, subtract_mean, mean_image,
            use_rms_noise_estimator, boxsize, selfnoise_offset, dde_noise, ddenoise_scaling_factor, full_bandwidth,
            std::nullopt, std::nullopt, reject_noise_threshold, histogram_detail, max_sigma_bin, threshold);
      }

      if (result_xtensor.processed && result_manual.processed)
      {
        ++processed_count;
        const xt::xarray<float> snr_diff = xt::abs(result_xtensor.snr - result_manual.snr);
        const xt::xarray<float> noisemap_diff = xt::abs(result_xtensor.noisemap - result_manual.noisemap);
        max_snr_diff = std::max(max_snr_diff, static_cast<float>(xt::amax(snr_diff)()));
        max_noisemap_diff = std::max(max_noisemap_diff, static_cast<float>(xt::amax(noisemap_diff)()));
      }
    }

    ScopedTimer::report();
    std::printf("  processed=%d/%d, max |snr| diff=%.6g, max |noisemap| diff=%.6g\n", processed_count, iterations,
               max_snr_diff, max_noisemap_diff);
  }

  return 0;
}

// Builds one synthetic (interval, stokes) item plus the single PlaneBatchGroup all items in this
// benchmark share (accept_mask/mean_image/selfnoise/dde_noise don't vary across the synthetic
// batch here, same as a real single-Stokes-label group would look like).
PlaneBatchItem make_plane_item(std::mt19937 &rng, size_t npix, int num_sources, size_t group_index)
{
  PlaneBatchItem item;
  item.group_index = group_index;

  // Background noise matching the rough magnitude of real per-pixel flux values, same as the band
  // benchmark above -- the exact distribution doesn't matter for a hotspot/scaling profile, only
  // the array shape/dtype does.
  item.stokes_cube = random_array(rng, {1, npix, npix}, 0.f, 1e-4f);

  // A handful of bright point sources well above the detection threshold, so label_islands()/
  // find_island_peak()/compute_adjusted_noise_stats() actually get exercised -- a plane with no
  // detections at all would skip the per-island code entirely, giving an unrealistic profile.
  std::uniform_int_distribution<size_t> pix_dist(0, npix - 1);
  for (int i = 0; i < num_sources; ++i)
  {
    item.stokes_cube(0, pix_dist(rng), pix_dist(rng)) = 0.01f;
  }

  item.weights = xt::xarray<float>({1});
  item.weights(0) = 1.0f;
  item.nonzero_channels = xt::xarray<bool>(std::vector<size_t>{1});
  item.nonzero_channels(0) = true;
  return item;
}

PlaneBatchGroup make_plane_group(std::mt19937 &rng, size_t npix)
{
  PlaneBatchGroup group;
  group.accept_mask = xt::xarray<bool>(std::vector<size_t>{1, npix, npix});
  for (auto &v : group.accept_mask)
  {
    v = true;
  }
  group.mean_image = random_array(rng, {1, npix, npix}, 0.f, 1e-5f);
  // selfnoise/dde_noise default to zero when unavailable in the real pipeline, same as the band
  // benchmark above.
  group.selfnoise_offset = xt::zeros<float>({npix, npix});
  group.selfnoise_raw = group.selfnoise_offset;
  group.dde_noise = xt::zeros<float>({npix, npix});
  return group;
}

// Parameters matching the values used in the manual `breifast procres` timing runs (boxsize=400,
// threshold=6.0, reject-noise-threshold=10, no flux/snr cutoff images, no beam fitting), same as
// run_band_slice_bench() above.
PlaneConfig make_plane_config()
{
  PlaneConfig config;
  config.band_slices = {BandSlice{0, 1, "B0"}};
  config.subtract_mean = true;
  config.use_rms_noise_estimator = true;
  config.boxsize = 400;
  config.inner_boxsize = 10;
  config.ddenoise_scaling_factor = 1.0f;
  config.full_bandwidth = 763.91e6f;
  config.reject_noise_threshold = 10.0f;
  config.histogram_detail = 0;
  config.max_sigma_bin = 200;
  config.threshold = 6.0f;
  config.save_stat_cubes = false;
  config.beam_size_enabled = false;
  return config;
}

int run_plane_bench(int argc, char **argv)
{
  const int iterations = argc > 0 ? std::atoi(argv[0]) : 5;
  const size_t npix = argc > 1 ? static_cast<size_t>(std::atoi(argv[1])) : 3072;
  const int num_items = argc > 2 ? std::atoi(argv[2]) : 8;
  const int num_threads = argc > 3 ? std::atoi(argv[3]) : 0;

  if (num_threads > 0)
  {
    std::printf("bench plane: %d iterations, %zux%zu plane, %d items/batch, num_threads=%d\n", iterations, npix,
               npix, num_items, num_threads);
  }
  else
  {
    std::printf("bench plane: %d iterations, %zux%zu plane, %d items/batch, num_threads=default\n", iterations,
               npix, npix, num_items);
  }

  const PlaneConfig config = make_plane_config();
  xt::xarray<float> weights_template({1});
  weights_template(0) = 1.0f;
  const xt::xarray<float> channel_width = xt::full_like(weights_template, 763.91e6f);
  std::mt19937 rng(42);

  double sequential_total = 0.0;
  double batch_total = 0.0;

  for (int iter = 0; iter < iterations; ++iter)
  {
    // Build a fresh batch of items each iteration -- process_plane_core() mutates stokes_cube in
    // place, so the sequential and batch runs each need their own untouched copy of the same input
    // to stay comparable.
    const PlaneBatchGroup group = make_plane_group(rng, npix);
    std::vector<PlaneBatchItem> seq_items;
    seq_items.reserve(static_cast<size_t>(num_items));
    for (int i = 0; i < num_items; ++i)
    {
      seq_items.push_back(make_plane_item(rng, npix, 5, 0));
    }
    std::vector<PlaneBatchItem> batch_items = seq_items;

    const auto t0 = std::chrono::steady_clock::now();
    int seq_detections = 0;
    for (auto &item : seq_items)
    {
      ProcessPlaneResult result =
          process_plane_core(item.stokes_cube, item.weights, group.accept_mask, channel_width, item.nonzero_channels,
                             group.mean_image, group.selfnoise_offset, group.selfnoise_raw, group.dde_noise,
                             group.flux_cutoff, group.snr_cutoff, std::nullopt, item.bmaj, item.bmin, item.bpa,
                             config);
      seq_detections += static_cast<int>(result.detections.x.size());
    }
    const auto t1 = std::chrono::steady_clock::now();

    // The actual batch dispatch (tbb::parallel_for, optional thread cap, per-item try/catch) --
    // process_plane_batch()'s pybind11 wrapper calls this exact same function.
    const std::vector<PlaneBatchGroup> groups = {group};
    std::vector<PlaneBatchItemResult> batch_results =
        process_plane_batch_core(batch_items, groups, channel_width, std::nullopt, config, num_threads);
    const auto t2 = std::chrono::steady_clock::now();

    const double seq_seconds = std::chrono::duration<double>(t1 - t0).count();
    const double batch_seconds = std::chrono::duration<double>(t2 - t1).count();
    sequential_total += seq_seconds;
    batch_total += batch_seconds;

    int batch_detections = 0;
    for (const auto &r : batch_results)
    {
      if (!r.has_error)
      {
        batch_detections += static_cast<int>(r.result.detections.x.size());
      }
    }

    std::printf("  iter %2d: sequential=%.3fs (%d detections)  batch=%.3fs (%d detections)  speedup=%.2fx\n", iter,
                seq_seconds, seq_detections, batch_seconds, batch_detections,
                batch_seconds > 0 ? seq_seconds / batch_seconds : 0.0);
  }

  std::printf("\nTotals over %d iterations: sequential=%.3fs  batch=%.3fs  speedup=%.2fx\n", iterations,
              sequential_total, batch_total, batch_total > 0 ? sequential_total / batch_total : 0.0);

  return 0;
}

} // namespace

int main(int argc, char **argv)
{
  if (argc < 2 || (std::strcmp(argv[1], "band") != 0 && std::strcmp(argv[1], "plane") != 0))
  {
    std::fprintf(stderr, "usage: %s <band|plane> [args...]\n", argv[0]);
    return 1;
  }

  if (std::strcmp(argv[1], "band") == 0)
  {
    return run_band_slice_bench(argc - 2, argv + 2);
  }
  return run_plane_bench(argc - 2, argv + 2);
}
