#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <optional>

#include "gpu_plane_integration.hpp"

using namespace breifast;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("gpu_process_plane_batch produces the expected plane result", "[gpu][integration]")
{
  constexpr std::size_t n_y = 3;
  constexpr std::size_t n_x = 4;
  constexpr std::size_t n_channels = 2;

  PlaneBatchItem item;
  item.stokes_cube.resize({n_channels, n_y, n_x});
  item.stokes_cube.fill(1.F);
  item.weights = xt::xarray<float>({1.F, 1.F});
  item.nonzero_channels = xt::xarray<bool>({true, true});
  item.stokes_label = "I";

  PlaneBatchGroup group;
  group.accept_mask.resize({n_channels, n_y, n_x});
  group.accept_mask.fill(false);
  group.accept_mask(0, 0, 0) = true;
  group.selfnoise_offset.resize({n_y, n_x});
  group.selfnoise_offset.fill(0.F);
  group.selfnoise_raw.resize({n_y, n_x});
  group.selfnoise_raw.fill(1.F);
  group.dde_noise.resize({n_y, n_x});
  group.dde_noise.fill(0.F);

  PlaneConfig config;
  config.subtract_mean = false;
  config.use_rms_noise_estimator = true;
  config.boxsize = 3;
  config.inner_boxsize = 0;
  config.histogram_detail = 1;
  config.max_sigma_bin = 5;
  config.threshold = 0.5F;
  config.beam_size_enabled = false;
  config.band_slices = {BandSlice{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "band"}};

  const double expected_extended_box_snr = static_cast<double>(config.max_to_std_ratio_value());
  const auto result = gpu::gpu_process_plane_batch(item, group, config);

  CHECK(result.any_data == false);
  REQUIRE(result.detections.size() == 1);
  CHECK(result.detections.stokes_label()[0] == "I");
  CHECK(result.detections.band_label()[0] == "band");
  CHECK(result.detections.peak_y()[0] == 0);
  CHECK(result.detections.peak_x()[0] == 0);
  CHECK(result.detections.island_size()[0] == 1);
  CHECK_THAT(result.detections.flux()[0], WithinRel(1.F, 1e-6));
  CHECK_THAT(result.detections.local_noise()[0], WithinRel(1.F, 1e-6));
  CHECK_THAT(result.detections.local_noise_median_ratio()[0], WithinRel(1.F, 1e-6));
  CHECK_THAT(result.detections.signal_to_noise()[0], WithinRel(1.F, 1e-6));
  CHECK_THAT(result.detections.signal_to_noise_rms()[0], WithinAbs(0.F, 1e-6));
  CHECK_THAT(result.detections.signal_to_noise_extended_box()[0], WithinRel(expected_extended_box_snr, 1e-6));
  CHECK_THAT(result.detections.signal_to_noise_local_mad()[0], WithinRel(1.0 / 1.4826, 1e-6));
  CHECK_THAT(result.detections.signal_to_noise_global_mad()[0], WithinRel(1.F / 1.4826F, 1e-6));
  CHECK_THAT(result.detections.self_noise()[0], WithinRel(1.F, 1e-6));
  CHECK_THAT(result.detections.excess_noise_factor()[0], WithinRel(1.F, 1e-6));
  CHECK_THAT(result.detections.beam_chi2()[0], WithinAbs(0.F, 1e-6));
  CHECK_THAT(result.detections.boost()[0], WithinAbs(0.F, 1e-6));

  REQUIRE(result.min_sigma.has_value() == true);
  REQUIRE(result.max_sigma.has_value() == true);
  CHECK_THAT(*result.min_sigma, WithinRel(1.F, 1e-6));
  CHECK_THAT(*result.max_sigma, WithinRel(1.F, 1e-6));
  CHECK(result.histograms[2][6] == 1L);
  REQUIRE(result.median_noises.size() == 1);
  CHECK_THAT(result.median_noises[0], WithinRel(1.0, 1e-9));

  REQUIRE(result.variance_stats.has_value() == true);
  const auto &variance = *result.variance_stats;
  REQUIRE(variance.sum_pow2.shape(0) == n_y);
  REQUIRE(variance.sum_pow2.shape(1) == n_x);
  for (std::size_t i = 0; i < variance.sum_pow2.size(); ++i)
  {
    CHECK_THAT(variance.sum_pow2.data()[i], WithinRel(2.0, 1e-6));
    CHECK_THAT(variance.sum_pow3.data()[i], WithinRel(2.0, 1e-6));
    CHECK_THAT(variance.sum_pow4.data()[i], WithinRel(2.0, 1e-6));
  }
  REQUIRE(variance.nonzero_channels_mask.shape(0) == n_channels);
  CHECK(variance.nonzero_channels_mask(0) == true);
  CHECK(variance.nonzero_channels_mask(1) == true);
}
