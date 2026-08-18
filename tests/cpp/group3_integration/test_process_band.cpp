#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "plane_lot_dispatch.hpp"

#include "../test_helpers.hpp"

using namespace breifast;

TEST_CASE("process_band detects a single clear point-source island with correct identity and basic detection "
         "fields",
         "[process_band]")
{
  // 7x7 plane, single-channel band: background = 1.0 everywhere except the center pixel (3,3) = 50.0. The peak is a
  // 1-in-49 outlier, so it can never move the median -- keeping noise_global_mad trivially hand-computable.
  auto plane_rows = std::vector<std::vector<float>>(7, std::vector<float>(7, 1.F));
  plane_rows[3][3] = 50.F;
  xt::xarray<float> plane_2d;
  plane_2d.resize({7, 7});
  for (size_t i_y = 0; i_y < 7; ++i_y)
  {
    for (size_t i_x = 0; i_x < 7; ++i_x)
    {
      plane_2d(i_y, i_x) = plane_rows[i_y][i_x];
    }
  }

  PlaneBatchItem item;
  item.stokes_cube.resize({1, 7, 7});
  xt::view(item.stokes_cube, 0) = plane_2d;
  item.weights = xt::xarray<float>({1.F});
  item.nonzero_channels = xt::xarray<bool>({true});
  item.stokes_label = "I";

  PlaneBatchGroup group;
  group.accept_mask.resize({1, 7, 7});
  group.accept_mask.fill(true);
  group.selfnoise_offset.resize({7, 7});
  group.selfnoise_offset.fill(0.F);
  group.selfnoise_raw.resize({7, 7});
  group.selfnoise_raw.fill(2.5F);
  group.dde_noise.resize({7, 7});
  group.dde_noise.fill(0.F);

  BandSlice band{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "bandA"};

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = false; // Global-MAD noise -> a single broadcast constant, trivial to hand-verify;
                                          // the RMS-box estimator is already covered directly by
                                          // compute_rms_noise_map's own tests.
  config.boxsize = 3;
  config.inner_boxsize = 0; // Zeroes only the postage stamp's own center pixel, leaving its 8-neighbor donut intact.
  config.reject_noise_threshold = 0.F;
  config.threshold = 5.F; // Background snr ~0.67 fails this; the center's snr ~33.7 clears it -- exactly one
                          // size-1 island.
  config.beam_size_enabled = false;

  ProcessPlaneResult result;
  process_band(item, group, band, config, result);

  REQUIRE(result.detections.peak_y.size() == 1);
  CHECK(result.detections.peak_y[0] == 3);
  CHECK(result.detections.peak_x[0] == 3);
  CHECK(result.detections.island_size[0] == 1);
  CHECK(result.detections.stokes_label[0] == "I");
  CHECK(result.detections.band_label[0] == "bandA");

  // noise_global_mad = median(|plane|) * 1.4826 = 1.0 * 1.4826, since only 1 of 49 pixels differs from 1.0.
  const float expected_noise = 1.4826F;
  CHECK(result.detections.local_noise[0] == Catch::Approx(expected_noise).epsilon(1e-4));
  CHECK(result.detections.local_noise_median_ratio[0] == Catch::Approx(1.F).epsilon(1e-4));

  const float expected_snr = 50.F / expected_noise;
  CHECK(result.detections.signal_to_noise[0] == Catch::Approx(expected_snr).epsilon(1e-4));
  // Uniform background -> the postage stamp's 8-neighbor donut is all identical values, so its stddev is exactly 0
  // and signal_to_noise_rms hits the rms==0 guard.
  CHECK(result.detections.signal_to_noise_rms[0] == 0.F);
  CHECK(result.detections.signal_to_noise_local_mad[0] == Catch::Approx(expected_snr).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_global_mad[0] == Catch::Approx(expected_snr).epsilon(1e-4));

  // signal_to_noise_extended_box = flux * max_to_std_ratio_value(): computed via the same config call the
  // production code uses, rather than hardcoding a literal -- that ratio is itself the output of a numerical
  // search Group 1's max_to_std_ratio tests already pin down independently.
  const float expected_extended_box = 50.F * config.max_to_std_ratio_value();
  CHECK(result.detections.signal_to_noise_extended_box[0] == Catch::Approx(expected_extended_box).epsilon(1e-4));

  CHECK(result.detections.self_noise[0] == Catch::Approx(2.5F));
  CHECK(result.detections.excess_noise_factor[0] == Catch::Approx(1.F).epsilon(1e-4));
  CHECK(result.detections.beam_chi2[0] == 0.F);
  CHECK(result.detections.boost[0] == 0.F);
}

TEST_CASE("process_band zero-pads postage stamps correctly for islands touching all four plane edges",
         "[process_band]")
{
  // 5x5 plane, boxsize=3 (radius=1, pad=2). Top-left peak (0,0)=50 pulls its postage stamp's top/left edges into
  // padded index 1 (< pad) -- the duplicate-condition bug fixed earlier. Bottom-right peak (4,4)=60 pulls its
  // stamp's bottom/right edges into padded index 7 (== pad+n_y == pad+n_x) -- the off-by-one bug fixed just now.
  // Both islands share the same decoy neighbor set {2,4,6} by design, so their noise-derived stats are directly
  // comparable and any padding regression (wrapped/garbage data instead of zeros) would corrupt both differently.
  xt::xarray<float> plane_2d;
  plane_2d.resize({5, 5});
  plane_2d.fill(1.F);
  plane_2d(0, 0) = 50.F; // top-left peak
  plane_2d(0, 1) = 2.F;
  plane_2d(1, 0) = 4.F;
  plane_2d(1, 1) = 6.F;
  plane_2d(4, 4) = 60.F; // bottom-right peak
  plane_2d(4, 3) = 6.F;
  plane_2d(3, 4) = 4.F;
  plane_2d(3, 3) = 2.F;

  PlaneBatchItem item;
  item.stokes_cube.resize({1, 5, 5});
  xt::view(item.stokes_cube, 0) = plane_2d;
  item.weights = xt::xarray<float>({1.F});
  item.nonzero_channels = xt::xarray<bool>({true});
  item.stokes_label = "I";

  PlaneBatchGroup group;
  group.accept_mask.resize({1, 5, 5});
  group.accept_mask.fill(true);
  group.selfnoise_offset.resize({5, 5});
  group.selfnoise_offset.fill(0.F);
  group.selfnoise_raw.resize({5, 5});
  group.selfnoise_raw.fill(2.5F);
  group.selfnoise_raw(0, 0) = 1.1F; // distinct per-peak values, to prove each detection reads its own pixel.
  group.selfnoise_raw(4, 4) = 9.9F;
  group.dde_noise.resize({5, 5});
  group.dde_noise.fill(0.F);

  BandSlice band{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "bandA"};

  PlaneConfig config;
  config.use_rms_noise_estimator = false;
  config.boxsize = 3;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  // Peaks (snr ~33.7, ~40.5) clear this; every decoy/background value (max snr ~4.05) doesn't -- exactly two
  // size-1 islands.
  config.threshold = 5.F;
  config.beam_size_enabled = false;

  ProcessPlaneResult result;
  process_band(item, group, band, config, result);

  REQUIRE(result.detections.peak_y.size() == 2);
  // Detection order isn't guaranteed (islands come from an unordered_map), so look each one up by its peak
  // coordinates rather than assuming index 0/1.
  const auto find_index = [&](long peak_y, long peak_x)
  {
    for (size_t i = 0; i < result.detections.peak_y.size(); ++i)
    {
      if (result.detections.peak_y[i] == peak_y && result.detections.peak_x[i] == peak_x)
      {
        return i;
      }
    }
    FAIL("no detection at (" << peak_y << ", " << peak_x << ")");
    return static_cast<size_t>(0);
  };
  const auto tl = find_index(0, 0);
  const auto br = find_index(4, 4);

  const float expected_noise = 1.4826F;
  const float expected_stddev = 1.632993F; // stddev of the shared decoy donut {2, 4, 6}.
  const float expected_lmad = 5.9304F;     // median({2,4,6}) * 1.4826.
  const auto ratio = config.max_to_std_ratio_value();

  CHECK(result.detections.island_size[tl] == 1);
  CHECK(result.detections.local_noise[tl] == Catch::Approx(expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise[tl] == Catch::Approx(50.F / expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_rms[tl] == Catch::Approx(50.F / expected_stddev).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_local_mad[tl] == Catch::Approx(50.F / expected_lmad).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_global_mad[tl] == Catch::Approx(50.F / expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_extended_box[tl] == Catch::Approx(50.F * ratio / 6.F).epsilon(1e-4));
  CHECK(result.detections.self_noise[tl] == Catch::Approx(1.1F));

  CHECK(result.detections.island_size[br] == 1);
  CHECK(result.detections.local_noise[br] == Catch::Approx(expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise[br] == Catch::Approx(60.F / expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_rms[br] == Catch::Approx(60.F / expected_stddev).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_local_mad[br] == Catch::Approx(60.F / expected_lmad).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_global_mad[br] == Catch::Approx(60.F / expected_noise).epsilon(1e-4));
  CHECK(result.detections.signal_to_noise_extended_box[br] == Catch::Approx(60.F * ratio / 6.F).epsilon(1e-4));
  CHECK(result.detections.self_noise[br] == Catch::Approx(9.9F));
}

TEST_CASE("process_band computes a weighted, unequally-weighted multi-channel mean, with subtract_mean off and on",
         "[process_band]")
{
  // 5x5 plane, 2-channel band, weights=[3,1] (deliberately unequal -- a naive unweighted mean would give 40 at the
  // peak instead of the correct weighted 50, so this directly catches that bug class). Peak away from any edge, so
  // its postage stamp never touches padding (already covered by the border test above).
  xt::xarray<float> stokes_cube;
  stokes_cube.resize({2, 5, 5});
  xt::view(stokes_cube, 0) = 1.F;
  xt::view(stokes_cube, 1) = 1.F;
  stokes_cube(0, 2, 2) = 60.F;
  stokes_cube(1, 2, 2) = 20.F; // weighted mean at the peak: (3*60 + 1*20) / 4 = 50.

  PlaneBatchItem item;
  item.stokes_cube = stokes_cube;
  item.weights = xt::xarray<float>({3.F, 1.F});
  item.nonzero_channels = xt::xarray<bool>({true, true});
  item.stokes_label = "I";

  PlaneBatchGroup group;
  group.accept_mask.resize({2, 5, 5});
  group.accept_mask.fill(true);
  group.selfnoise_offset.resize({5, 5});
  group.selfnoise_offset.fill(0.F);
  group.selfnoise_raw.resize({5, 5});
  group.selfnoise_raw.fill(2.5F);
  group.dde_noise.resize({5, 5});
  group.dde_noise.fill(0.F);

  BandSlice band{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "bandA"};

  PlaneConfig config;
  config.use_rms_noise_estimator = false;
  config.boxsize = 3;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  config.threshold = 5.F;
  config.beam_size_enabled = false;

  const auto ratio = config.max_to_std_ratio_value();
  const float expected_noise = 1.4826F; // median(|plane|) is unmoved by a 1-in-25 outlier, in both parts below.

  // Part A: subtract_mean off.
  {
    config.subtract_mean = false;
    ProcessPlaneResult result;
    process_band(item, group, band, config, result);

    REQUIRE(result.detections.peak_y.size() == 1);
    CHECK(result.detections.peak_y[0] == 2);
    CHECK(result.detections.peak_x[0] == 2);
    CHECK(result.detections.signal_to_noise[0] == Catch::Approx(50.F / expected_noise).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise_global_mad[0] == Catch::Approx(50.F / expected_noise).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise_rms[0] == 0.F); // uniform-1.0 donut, same as the first test's case.
    CHECK(result.detections.signal_to_noise_extended_box[0] == Catch::Approx(50.F * ratio).epsilon(1e-4));
  }

  // Part B: subtract_mean on. mean_image = 4.0 on channel 0, 0.0 on channel 1, uniform over the grid,
  // inv_band_size=0.5 -> subtracts exactly 0.5*(4+0)=2.0 at every pixel: background 1.0 -> -1.0, peak 50 -> 48. A
  // clean, exact delta from part A isolates the subtraction's effect from everything else (noise stays 1.4826,
  // since abs() doesn't care about the sign flip on the background).
  {
    xt::xarray<float> mean_image;
    mean_image.resize({2, 5, 5});
    xt::view(mean_image, 0) = 4.F;
    xt::view(mean_image, 1) = 0.F;
    group.mean_image = mean_image;
    config.subtract_mean = true;
    ProcessPlaneResult result;
    process_band(item, group, band, config, result);

    REQUIRE(result.detections.peak_y.size() == 1);
    CHECK(result.detections.peak_y[0] == 2);
    CHECK(result.detections.peak_x[0] == 2);
    CHECK(result.detections.signal_to_noise[0] == Catch::Approx(48.F / expected_noise).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise_global_mad[0] == Catch::Approx(48.F / expected_noise).epsilon(1e-4));
    CHECK(result.detections.signal_to_noise_rms[0] == 0.F); // uniform-(-1.0) donut: still identical, still 0.
    CHECK(result.detections.signal_to_noise_extended_box[0] == Catch::Approx(48.F * ratio).epsilon(1e-4));
  }
}
