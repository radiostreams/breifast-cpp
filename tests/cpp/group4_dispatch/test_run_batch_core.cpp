#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "plane_lot_dispatch.hpp"

using namespace breifast;

namespace
{
PlaneBatchGroup make_group(size_t n, float selfnoise_raw_value)
{
  PlaneBatchGroup group;
  group.accept_mask.resize({1, n, n});
  group.accept_mask.fill(true);
  group.selfnoise_offset.resize({n, n});
  group.selfnoise_offset.fill(0.F);
  group.selfnoise_raw.resize({n, n});
  group.selfnoise_raw.fill(selfnoise_raw_value);
  group.dde_noise.resize({n, n});
  group.dde_noise.fill(0.F);
  return group;
}

PlaneBatchItem make_item(size_t n, long peak_y, long peak_x, size_t group_index)
{
  xt::xarray<float> plane;
  plane.resize({n, n});
  plane.fill(1.F);
  plane(static_cast<size_t>(peak_y), static_cast<size_t>(peak_x)) = 60.F;

  PlaneBatchItem item;
  item.stokes_cube.resize({1, n, n});
  xt::view(item.stokes_cube, 0) = plane;
  item.weights = xt::xarray<float>({1.F});
  item.nonzero_channels = xt::xarray<bool>({true});
  item.stokes_label = "I";
  item.group_index = group_index;
  return item;
}
} // namespace

TEST_CASE("run_batch_core resolves each item's group_index into the correct group and preserves items order",
         "[run_batch_core]")
{
  // Uniform background=1.0 with one peak=60.0 gives median(abs(plane))=1.0, so
  // noise_global_mad=1.4826 -- background snr~=0.67, peak snr~=40.5, so threshold=4.0 yields exactly
  // one size-1 island per item, regardless of which group it's paired with.
  constexpr size_t n = 7;
  std::vector<PlaneBatchGroup> groups{make_group(n, 1.F), make_group(n, 2.F)};

  std::vector<PlaneBatchItem> items{
      make_item(n, 2, 2, 0),
      make_item(n, 4, 4, 1),
      make_item(n, 2, 4, 0),
  };

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = false;
  config.boxsize = 3;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  config.threshold = 4.F;
  config.beam_size_enabled = false;
  config.band_slices = {BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"}};

  const auto results = run_batch_core(items, groups, config, 2);

  REQUIRE(results.size() == 3);

  REQUIRE(results[0].detections.peak_y().size() == 1);
  CHECK(results[0].detections.peak_y()[0] == 2);
  CHECK(results[0].detections.peak_x()[0] == 2);
  CHECK(results[0].detections.self_noise()[0] == Catch::Approx(1.F));

  REQUIRE(results[1].detections.peak_y().size() == 1);
  CHECK(results[1].detections.peak_y()[0] == 4);
  CHECK(results[1].detections.peak_x()[0] == 4);
  CHECK(results[1].detections.self_noise()[0] == Catch::Approx(2.F));

  REQUIRE(results[2].detections.peak_y().size() == 1);
  CHECK(results[2].detections.peak_y()[0] == 2);
  CHECK(results[2].detections.peak_x()[0] == 4);
  CHECK(results[2].detections.self_noise()[0] == Catch::Approx(1.F));
}

TEST_CASE("run_batch_core keeps results in items order across many items scheduled on multiple TBB threads",
         "[run_batch_core]")
{
  constexpr size_t n = 9;
  constexpr size_t num_items = 40;
  constexpr size_t num_groups = 4;

  std::vector<PlaneBatchGroup> groups;
  for (size_t i = 0; i < num_groups; ++i)
  {
    groups.push_back(make_group(n, static_cast<float>(i) + 1.F));
  }

  std::vector<PlaneBatchItem> items;
  std::vector<std::pair<long, long>> expected_peaks;
  for (size_t i = 0; i < num_items; ++i)
  {
    const auto peak_y = static_cast<long>(i / n);
    const auto peak_x = static_cast<long>(i % n);
    expected_peaks.emplace_back(peak_y, peak_x);
    items.push_back(make_item(n, peak_y, peak_x, i % num_groups));
  }

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = false;
  config.boxsize = 3;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  config.threshold = 4.F;
  config.beam_size_enabled = false;
  config.band_slices = {BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"}};

  const auto results = run_batch_core(items, groups, config, 4);

  REQUIRE(results.size() == num_items);
  for (size_t i = 0; i < num_items; ++i)
  {
    INFO("item " << i);
    REQUIRE(results[i].detections.peak_y().size() == 1);
    CHECK(results[i].detections.peak_y()[0] == expected_peaks[i].first);
    CHECK(results[i].detections.peak_x()[0] == expected_peaks[i].second);
    CHECK(results[i].detections.self_noise()[0] == Catch::Approx(static_cast<float>(i % num_groups) + 1.F));
  }
}

TEST_CASE("run_batch_core isolates a per-item exception: one item throwing doesn't abort the batch or corrupt "
         "other items' results",
         "[run_batch_core]")
{
  // group 1's selfnoise_offset is deliberately the wrong shape (3x3 instead of item1's actual 7x7 plane), so
  // process_band's noisemap = sqrt(noisemap0^2 + scale*selfnoise_offset + ...) throws xtensor's
  // "Incompatible dimension of arrays" broadcast error for item 1 only.
  constexpr size_t n = 7;
  auto broken_group = make_group(n, 9.F);
  broken_group.selfnoise_offset.resize({3, 3});
  broken_group.selfnoise_offset.fill(0.F);
  std::vector<PlaneBatchGroup> groups{make_group(n, 1.F), broken_group};

  std::vector<PlaneBatchItem> items{
      make_item(n, 2, 2, 0),
      make_item(n, 4, 4, 1),
      make_item(n, 2, 4, 0),
  };

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = false;
  config.boxsize = 3;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  config.threshold = 4.F;
  config.beam_size_enabled = false;
  config.band_slices = {BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"}};

  const auto results = run_batch_core(items, groups, config, 2);

  REQUIRE(results.size() == 3);

  REQUIRE(results[0].detections.peak_y().size() == 1);
  CHECK(results[0].detections.peak_y()[0] == 2);
  CHECK(results[0].detections.peak_x()[0] == 2);

  CHECK(results[1].detections.peak_y().empty());

  REQUIRE(results[2].detections.peak_y().size() == 1);
  CHECK(results[2].detections.peak_y()[0] == 2);
  CHECK(results[2].detections.peak_x()[0] == 4);
}

TEST_CASE("run_lot_core forwards lot.planes/lot.groups to run_batch_core unchanged", "[run_lot_core]")
{
  constexpr size_t n = 7;

  LotBatch lot;
  lot.groups = {make_group(n, 1.F), make_group(n, 2.F)};
  lot.planes = {
      make_item(n, 2, 2, 0),
      make_item(n, 4, 4, 1),
  };

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = false;
  config.boxsize = 3;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  config.threshold = 4.F;
  config.beam_size_enabled = false;
  config.band_slices = {BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"}};

  const auto results = run_lot_core(lot, config, 2);

  REQUIRE(results.size() == 2);

  REQUIRE(results[0].detections.peak_y().size() == 1);
  CHECK(results[0].detections.peak_y()[0] == 2);
  CHECK(results[0].detections.peak_x()[0] == 2);
  CHECK(results[0].detections.self_noise()[0] == Catch::Approx(1.F));

  REQUIRE(results[1].detections.peak_y().size() == 1);
  CHECK(results[1].detections.peak_y()[0] == 4);
  CHECK(results[1].detections.peak_x()[0] == 4);
  CHECK(results[1].detections.self_noise()[0] == Catch::Approx(2.F));
}

TEST_CASE("run_batch_core processes multi-band items (band_slices on the shared PlaneConfig) in the same batch as "
         "single-band items",
         "[run_batch_core]")
{
  // Each item has its own peak position, present identically in both channels, so every band (B0, B1, and the
  // combined B0~1) detects the same island for that item -- isolates "does run_batch_core drive process_plane's
  // full multi-band loop per item" from process_plane's own band-accumulation logic (already covered by
  // test_process_plane.cpp).
  constexpr size_t n = 7;
  std::vector<PlaneBatchGroup> groups{make_group(n, 1.F)};

  const auto make_two_channel_item = [&](long peak_y, long peak_x)
  {
    xt::xarray<float> plane;
    plane.resize({n, n});
    plane.fill(1.F);
    plane(static_cast<size_t>(peak_y), static_cast<size_t>(peak_x)) = 60.F;

    PlaneBatchItem item;
    item.stokes_cube.resize({2, n, n});
    xt::view(item.stokes_cube, 0) = plane;
    xt::view(item.stokes_cube, 1) = plane;
    item.weights = xt::xarray<float>({1.F, 1.F});
    item.nonzero_channels = xt::xarray<bool>({true, true});
    item.stokes_label = "I";
    item.group_index = 0;
    return item;
  };

  std::vector<PlaneBatchItem> items{
      make_two_channel_item(2, 2),
      make_two_channel_item(4, 4),
  };

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = false;
  config.boxsize = 3;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  config.threshold = 4.F;
  config.beam_size_enabled = false;
  config.band_slices = {
      BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"},
      BandSlice{.start = 1, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "B1"},
      BandSlice{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "B0~1"},
  };

  const auto results = run_batch_core(items, groups, config, 2);

  REQUIRE(results.size() == 2);
  REQUIRE(results[0].detections.peak_y().size() == 3);
  REQUIRE(results[1].detections.peak_y().size() == 3);

  const auto has_band = [](const ProcessPlaneResult &result, const std::string &band_label, long peak_y, long peak_x)
  {
    for (size_t i = 0; i < result.detections.band_label().size(); ++i)
    {
      if (result.detections.band_label()[i] == band_label)
      {
        return result.detections.peak_y()[i] == peak_y && result.detections.peak_x()[i] == peak_x;
      }
    }
    return false;
  };

  CHECK(has_band(results[0], "B0", 2, 2));
  CHECK(has_band(results[0], "B1", 2, 2));
  CHECK(has_band(results[0], "B0~1", 2, 2));

  CHECK(has_band(results[1], "B0", 4, 4));
  CHECK(has_band(results[1], "B1", 4, 4));
  CHECK(has_band(results[1], "B0~1", 4, 4));
}
