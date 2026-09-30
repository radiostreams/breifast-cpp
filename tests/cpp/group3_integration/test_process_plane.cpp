#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "plane_lot_dispatch.hpp"

#include "../test_helpers.hpp"

using namespace breifast;

TEST_CASE("process_plane accumulates detections across all bands of a plane, processed in an order that doesn't "
         "let single-channel bands corrupt a later combined-channel band's flux",
         "[process_plane]")
{
  ProcessBandScratch scratch;
  // 7x7 plane, 2 channels, weights=[1,1]. channel0 background=1.0/peak=60, channel1 background=3.0/peak=40 (peak
  // at (3,3) in both). band_slices in the natural construction order breifast.py:517-524 uses (single-channel
  // bands first, then the combined band) -- Python itself iterates this list reversed (breifast.py:659,
  // `band_slices[::-1]`) specifically so process_band's single-channel snr-overwrite optimization
  // (plane_lot_dispatch.cpp:339-342) never corrupts a combined band's flux. Expected values below are from an
  // independent Python reference script re-deriving breifast.py's exact per-band formulas (global-MAD noise,
  // weighted mean, threshold, island peak selection) with that same reversed-order/accumulate-across-bands
  // behavior -- not from this C++ code.
  constexpr size_t n = 7;
  xt::xarray<float> channel0, channel1;
  channel0.resize({n, n});
  channel1.resize({n, n});
  channel0.fill(1.F);
  channel1.fill(3.F);
  channel0(3, 3) = 60.F;
  channel1(3, 3) = 40.F;

  PlaneBatchItem item;
  item.stokes_cube.resize({2, n, n});
  xt::view(item.stokes_cube, 0) = channel0;
  xt::view(item.stokes_cube, 1) = channel1;
  item.weights = xt::xarray<float>({1.F, 1.F});
  item.nonzero_channels = xt::xarray<bool>({true, true});
  item.stokes_label = "I";

  PlaneBatchGroup group;
  group.accept_mask.resize({2, n, n});
  group.accept_mask.fill(true);
  group.selfnoise_offset.resize({n, n});
  group.selfnoise_offset.fill(0.F);
  group.selfnoise_raw.resize({n, n});
  group.selfnoise_raw.fill(0.F);
  group.dde_noise.resize({n, n});
  group.dde_noise.fill(0.F);

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = false;
  config.boxsize = 3;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  config.threshold = 5.F;
  config.beam_size_enabled = false;
  config.band_slices = {
      BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"},
      BandSlice{.start = 1, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "B1"},
      BandSlice{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "B0~1"},
  };

  const auto result = process_plane(scratch, config, item, group);

  REQUIRE(result.detections.peak_y().size() == 3);
  const auto find_band = [&](const std::string &band_label)
  {
    for (size_t i = 0; i < result.detections.band_label().size(); ++i)
    {
      if (result.detections.band_label()[i] == band_label)
      {
        return i;
      }
    }
    FAIL("no detection for band " << band_label);
    return static_cast<size_t>(0);
  };

  const auto b0 = find_band("B0");
  CHECK(result.detections.peak_y()[b0] == 3);
  CHECK(result.detections.peak_x()[b0] == 3);
  CHECK(result.detections.island_size()[b0] == 1);
  CHECK(result.detections.signal_to_noise_global_mad()[b0] == Catch::Approx(40.46944556859571).epsilon(1e-4));

  const auto b1 = find_band("B1");
  CHECK(result.detections.peak_y()[b1] == 3);
  CHECK(result.detections.peak_x()[b1] == 3);
  CHECK(result.detections.island_size()[b1] == 1);
  CHECK(result.detections.signal_to_noise_global_mad()[b1] == Catch::Approx(8.993210126354603).epsilon(1e-4));

  const auto combined = find_band("B0~1");
  CHECK(result.detections.peak_y()[combined] == 3);
  CHECK(result.detections.peak_x()[combined] == 3);
  CHECK(result.detections.island_size()[combined] == 1);
  CHECK(result.detections.signal_to_noise_global_mad()[combined] == Catch::Approx(16.86226898691488).epsilon(1e-4));

  // One median_noises entry per process_band() call, in the same reversed (B0~1, B1, B0) order
  // process_plane actually processes bands in -- not construction order. Each value is
  // median(|plane|) * 1.4826 for that band's own weighted image (median(|plane|) = 1.0 for B0,
  // 3.0 for B1, 2.0 for the weighted-mean B0~1 -- cross-checked against the
  // signal_to_noise_global_mad values above, e.g. B0's flux 60 / 1.4826 == 40.469...).
  REQUIRE(result.median_noises.size() == 3);
  CHECK(result.median_noises[0] == Catch::Approx(2.9652).epsilon(1e-4));  // B0~1
  CHECK(result.median_noises[1] == Catch::Approx(4.4478).epsilon(1e-4));  // B1
  CHECK(result.median_noises[2] == Catch::Approx(1.4826).epsilon(1e-4));  // B0
}

TEST_CASE("process_plane's histograms reflect only the last band processed, not accumulated across bands",
         "[process_plane]")
{
  ProcessBandScratch scratch;
  // 2x2 plane, 2 single-channel bands. channel0 background=1.0/outlier=5.0, channel1 background=2.0/outlier=8.0.
  // Both give the same background bin (background/noise_global_mad is always 1/1.4826, regardless of scale), but a
  // different outlier bin, so a leaked/accumulated earlier band would show up as an extra populated bin.
  // band_slices=[B0,B1] in construction order; the reversed processing order (see the test above) processes B1
  // first, B0 last, so the final histograms should reflect only B0.
  xt::xarray<float> channel0, channel1;
  channel0.resize({2, 2});
  channel1.resize({2, 2});
  channel0.fill(1.F);
  channel1.fill(2.F);
  channel0(1, 1) = 5.F;
  channel1(1, 1) = 8.F;

  PlaneBatchItem item;
  item.stokes_cube.resize({2, 2, 2});
  xt::view(item.stokes_cube, 0) = channel0;
  xt::view(item.stokes_cube, 1) = channel1;
  item.weights = xt::xarray<float>({1.F, 1.F});
  item.nonzero_channels = xt::xarray<bool>({true, true});
  item.stokes_label = "I";

  PlaneBatchGroup group;
  group.accept_mask.resize({2, 2, 2});
  group.accept_mask.fill(true);
  group.selfnoise_offset.resize({2, 2});
  group.selfnoise_offset.fill(0.F);
  group.selfnoise_raw.resize({2, 2});
  group.selfnoise_raw.fill(0.F);
  group.dde_noise.resize({2, 2});
  group.dde_noise.fill(0.F);

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = false;
  config.boxsize = 1;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  config.threshold = 100.F; // no detections needed; only the histogram fields are under test
  config.beam_size_enabled = false;
  config.histogram_detail = 2;
  config.max_sigma_bin = 10;
  config.band_slices = {
      BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"},
      BandSlice{.start = 1, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "B1"},
  };

  const auto result = process_plane(scratch, config, item, group);

  // Bins {10: 3, 13: 1}, confirmed via an independent Python reference script.
  std::vector<long> expected(20, 0);
  expected[10] = 3;
  expected[13] = 1;
  CHECK(result.histograms[0] == expected);
  CHECK(result.histograms[1] == expected);
  CHECK(result.histograms[2] == expected);
}

TEST_CASE("process_plane only computes variance_stats for multi-channel items", "[process_plane]")
{
  ProcessBandScratch scratch;
  const auto make_group = [](size_t n)
  {
    PlaneBatchGroup group;
    group.accept_mask.resize({n, 2, 2});
    group.accept_mask.fill(true);
    group.selfnoise_offset.resize({2, 2});
    group.selfnoise_offset.fill(0.F);
    group.selfnoise_raw.resize({2, 2});
    group.selfnoise_raw.fill(0.F);
    group.dde_noise.resize({2, 2});
    group.dde_noise.fill(0.F);
    return group;
  };

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = false;
  config.boxsize = 1;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  config.threshold = 100.F;
  config.beam_size_enabled = false;

  // Part A: single-channel item -> no variance_stats.
  {
    PlaneBatchItem item;
    item.stokes_cube.resize({1, 2, 2});
    item.stokes_cube.fill(1.F);
    item.weights = xt::xarray<float>({1.F});
    item.nonzero_channels = xt::xarray<bool>({true});
    item.stokes_label = "I";

    PlaneConfig config_a = config;
    config_a.band_slices = {BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"}};

    const auto result = process_plane(scratch, config_a, item, make_group(1));
    CHECK_FALSE(result.variance_stats.has_value());
  }

  // Part B: multi-channel item -> variance_stats populated.
  {
    PlaneBatchItem item;
    item.stokes_cube.resize({2, 2, 2});
    item.stokes_cube.fill(1.F);
    item.weights = xt::xarray<float>({1.F, 1.F});
    item.nonzero_channels = xt::xarray<bool>({true, true});
    item.stokes_label = "I";

    PlaneConfig config_b = config;
    config_b.band_slices = {BandSlice{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "B0~1"}};

    const auto result = process_plane(scratch, config_b, item, make_group(2));
    CHECK(result.variance_stats.has_value());
  }
}

TEST_CASE("process_plane computes multi-channel variance_stats on the SNR cube left by the single-channel bands",
          "[process_plane]")
{
  // breifast.py writes each single-channel band's snr back into stokes_cube, then computes the spectral variance
  // stats from that cube "in units of SNR". Global-MAD noise with zero selfnoise/dde makes each channel's noise
  // median(|channel|) * 1.4826. Expected sums come from numpy on the same float32 inputs.
  ProcessBandScratch scratch;
  PlaneBatchGroup group;
  group.accept_mask.resize({2, 3, 3});
  group.accept_mask.fill(true);
  group.selfnoise_offset = xt::zeros<float>({3, 3});
  group.selfnoise_raw = xt::zeros<float>({3, 3});
  group.dde_noise = xt::zeros<float>({3, 3});

  PlaneBatchItem item;
  item.stokes_cube.resize({2, 3, 3});
  xt::view(item.stokes_cube, 0) = xt::xarray<float>({{1.F, -2.F, 3.F}, {-4.F, 5.F, -6.F}, {7.F, -8.F, 9.F}});
  xt::view(item.stokes_cube, 1) =
      xt::xarray<float>({{0.5F, 1.5F, -2.5F}, {3.5F, -4.5F, 5.5F}, {-6.5F, 7.5F, 8.5F}});
  item.weights = xt::xarray<float>({1.F, 1.F});
  item.nonzero_channels = xt::xarray<bool>({true, true});
  item.stokes_label = "I";

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = false;
  config.boxsize = 1;
  config.inner_boxsize = 0;
  config.reject_noise_threshold = 0.F;
  config.threshold = 100.F;
  config.beam_size_enabled = false;
  // Same order as breifast.py builds them: single channels first, aggregated band last (processed first).
  config.band_slices = {
      BandSlice{.start = 0, .stop = 1, .selfnoise_scale_factor = 1.F, .label = "B0"},
      BandSlice{.start = 1, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "B1"},
      BandSlice{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "B0~1"},
  };

  const auto result = process_plane(scratch, config, item, group);
  REQUIRE(result.variance_stats.has_value());
  const auto &stats = *result.variance_stats;

  const float expected_pow2[9] = {0.0238140281F, 0.123338692F, 0.304190487F, 0.566369414F, 0.909875572F,
                                  1.33470881F,   1.84086919F,  2.42835665F,  3.09717131F};
  const float expected_pow3[9] = {0.00287573156F, -0.00827362202F, 0.0136647634F, -0.0127320439F, 0.F,
                                  0.0300066471F,  -0.082763195F,   0.163744688F,  3.85754108F};
  const float expected_pow4[9] = {0.000362694729F, 0.00785355642F, 0.0465388782F, 0.160514385F, 0.413936764F,
                                  0.891023576F,    1.69605374F,    2.95336628F,   4.8073616F};
  for (size_t i = 0; i < 9; ++i)
  {
    INFO("pixel " << i);
    const size_t i_y = i / 3;
    const size_t i_x = i % 3;
    // Margin covers the center pixel of pow3, where the two channels cancel to float rounding noise.
    CHECK(stats.sum_pow2(i_y, i_x) == Catch::Approx(expected_pow2[i]).epsilon(1e-5).margin(1e-6));
    CHECK(stats.sum_pow3(i_y, i_x) == Catch::Approx(expected_pow3[i]).epsilon(1e-5).margin(1e-6));
    CHECK(stats.sum_pow4(i_y, i_x) == Catch::Approx(expected_pow4[i]).epsilon(1e-5).margin(1e-6));
  }
}
