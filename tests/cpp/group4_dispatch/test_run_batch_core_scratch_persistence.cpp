#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "plane_lot_dispatch.hpp"

using namespace breifast;
using Catch::Matchers::Equals;
using Catch::Matchers::RangeEquals;

namespace
{
PlaneBatchGroup make_group(size_t n_y, size_t n_x, float selfnoise_raw_value)
{
  PlaneBatchGroup group;
  group.accept_mask.resize({1, n_y, n_x});
  group.accept_mask.fill(true);
  group.selfnoise_offset.resize({n_y, n_x});
  group.selfnoise_offset.fill(0.F);
  group.selfnoise_raw.resize({n_y, n_x});
  group.selfnoise_raw.fill(selfnoise_raw_value);
  group.dde_noise.resize({n_y, n_x});
  group.dde_noise.fill(0.F);
  return group;
}

// Background noise in [-1, 1] plus `num_peaks` random-sign peaks of magnitude 20-80.
PlaneBatchItem make_random_item(size_t n_y, size_t n_x, size_t num_peaks, size_t group_index, std::mt19937 &rng)
{
  std::uniform_real_distribution<float> noise(-1.F, 1.F);
  std::uniform_real_distribution<float> magnitude(20.F, 80.F);
  std::uniform_int_distribution<size_t> position_y(0, n_y - 1);
  std::uniform_int_distribution<size_t> position_x(0, n_x - 1);
  std::bernoulli_distribution negative(0.5);

  PlaneBatchItem item;
  item.stokes_cube.resize({1, n_y, n_x});
  for (auto &value : item.stokes_cube)
  {
    value = noise(rng);
  }
  for (size_t i = 0; i < num_peaks; ++i)
  {
    const float peak = magnitude(rng);
    item.stokes_cube(0, position_y(rng), position_x(rng)) = negative(rng) ? -peak : peak;
  }
  item.weights = xt::xarray<float>({1.F});
  item.nonzero_channels = xt::xarray<bool>({true});
  item.stokes_label = "I";
  item.group_index = group_index;
  return item;
}

PlaneConfig make_config(bool use_rms_noise_estimator)
{
  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = use_rms_noise_estimator;
  config.boxsize = 5;
  config.inner_boxsize = 1;
  config.reject_noise_threshold = 0.F;
  config.threshold = 4.F;
  config.histogram_detail = 2;
  config.max_sigma_bin = 50;
  config.beam_size_enabled = false;
  config.band_slices = {BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"}};
  return config;
}

void check_same_result(const ProcessPlaneResult &actual, const ProcessPlaneResult &expected)
{
  const auto &a = actual.detections;
  const auto &e = expected.detections;
  CHECK_THAT(a.peak_y(), Equals(e.peak_y()));
  CHECK_THAT(a.peak_x(), Equals(e.peak_x()));
  CHECK_THAT(a.island_size(), Equals(e.island_size()));
  CHECK_THAT(a.stokes_label(), Equals(e.stokes_label()));
  CHECK_THAT(a.band_label(), Equals(e.band_label()));
  CHECK_THAT(a.flux(), Equals(e.flux()));
  CHECK_THAT(a.local_noise(), Equals(e.local_noise()));
  CHECK_THAT(a.local_noise_median_ratio(), Equals(e.local_noise_median_ratio()));
  CHECK_THAT(a.signal_to_noise(), Equals(e.signal_to_noise()));
  CHECK_THAT(a.signal_to_noise_rms(), Equals(e.signal_to_noise_rms()));
  CHECK_THAT(a.signal_to_noise_extended_box(), Equals(e.signal_to_noise_extended_box()));
  CHECK_THAT(a.signal_to_noise_local_mad(), Equals(e.signal_to_noise_local_mad()));
  CHECK_THAT(a.signal_to_noise_global_mad(), Equals(e.signal_to_noise_global_mad()));
  CHECK_THAT(a.self_noise(), Equals(e.self_noise()));
  CHECK_THAT(a.excess_noise_factor(), Equals(e.excess_noise_factor()));
  CHECK_THAT(a.beam_chi2(), Equals(e.beam_chi2()));
  CHECK_THAT(a.boost(), Equals(e.boost()));
  CHECK(actual.min_sigma == expected.min_sigma);
  CHECK(actual.max_sigma == expected.max_sigma);
  CHECK_THAT(actual.histograms, RangeEquals(expected.histograms));
  CHECK_THAT(actual.median_noises, Equals(expected.median_noises));
}
} // namespace

TEST_CASE("run_batch_core results are unaffected by scratch left over from a previous call with different inputs",
          "[run_batch_core][scratch]")
{
  // The per-thread scratch persists across calls, so call A leaves every buffer full of its own data (RMS noise
  // estimator, many bright peaks). Call B (global MAD, few faint-ish peaks) must match a reference computed item by
  // item with a brand-new ProcessBandScratch, which is guaranteed free of stale state.
  constexpr size_t n = 16;
  constexpr size_t num_items = 24;
  std::mt19937 rng(1234);
  std::vector<PlaneBatchGroup> groups{make_group(n, n, 1.F), make_group(n, n, 3.F)};

  std::vector<PlaneBatchItem> items_a;
  for (size_t i = 0; i < num_items; ++i)
  {
    items_a.push_back(make_random_item(n, n, 8, i % groups.size(), rng));
  }
  run_batch_core(items_a, groups, make_config(true), 4);

  std::vector<PlaneBatchItem> items_b;
  for (size_t i = 0; i < num_items; ++i)
  {
    items_b.push_back(make_random_item(n, n, 2, i % groups.size(), rng));
  }
  const auto config_b = make_config(false);

  std::vector<ProcessPlaneResult> expected;
  for (const auto &item : items_b)
  {
    ProcessBandScratch fresh_scratch;
    PlaneBatchItem item_copy = item;
    expected.push_back(process_plane(fresh_scratch, config_b, item_copy, groups[item.group_index]));
  }

  const auto results = run_batch_core(items_b, groups, config_b, 4);

  REQUIRE(results.size() == num_items);
  size_t total_detections = 0;
  for (size_t i = 0; i < num_items; ++i)
  {
    INFO("item " << i);
    check_same_result(results[i], expected[i]);
    total_detections += expected[i].detections.peak_y().size();
  }
  CHECK(total_detections > 0);
}

TEST_CASE("run_batch_core results are unaffected by scratch left over from previous calls with other plane shapes",
          "[run_batch_core][scratch]")
{
  // Shrink then grow, non-square both times, so every persistent buffer has to resize in both directions and a
  // transposed or stale shape would show up as a mismatch against the fresh-scratch reference.
  std::mt19937 rng(98765);
  const auto config = make_config(true);
  const std::vector<std::pair<size_t, size_t>> shapes{{24, 24}, {11, 19}, {21, 13}};

  for (const auto &[n_y, n_x] : shapes)
  {
    INFO("shape " << n_y << "x" << n_x);
    std::vector<PlaneBatchGroup> groups{make_group(n_y, n_x, 1.F), make_group(n_y, n_x, 2.F)};
    std::vector<PlaneBatchItem> items;
    for (size_t i = 0; i < 16; ++i)
    {
      items.push_back(make_random_item(n_y, n_x, 4, i % groups.size(), rng));
    }

    std::vector<ProcessPlaneResult> expected;
    for (const auto &item : items)
    {
      ProcessBandScratch fresh_scratch;
      PlaneBatchItem item_copy = item;
      expected.push_back(process_plane(fresh_scratch, config, item_copy, groups[item.group_index]));
    }

    const auto results = run_batch_core(items, groups, config, 4);

    REQUIRE(results.size() == items.size());
    size_t total_detections = 0;
    for (size_t i = 0; i < items.size(); ++i)
    {
      INFO("item " << i);
      check_same_result(results[i], expected[i]);
      total_detections += expected[i].detections.peak_y().size();
    }
    CHECK(total_detections > 0);
  }
}
