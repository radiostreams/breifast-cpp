#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <optional>

#include "gpu_band.hpp"
#include "gpu_plane.hpp"

using namespace breifast;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace
{
constexpr std::size_t n_y = 3;
constexpr std::size_t n_x = 4;
constexpr std::size_t n_channels = 2;

// A 2-channel plane of ones, 3x4 by default, accepted at (0, 0) only, which yields one size-1 island there.
struct Plane
{
  PlaneBatchItem item;
  PlaneBatchGroup group;
  PlaneConfig config;

  explicit Plane(std::size_t plane_n_y = n_y, std::size_t plane_n_x = n_x)
  {
    item.stokes_cube.resize({n_channels, plane_n_y, plane_n_x});
    item.stokes_cube.fill(1.F);
    item.weights = xt::xarray<float>({1.F, 1.F});
    item.nonzero_channels = xt::xarray<bool>({true, true});
    item.stokes_label = "I";

    group.accept_mask.resize({n_channels, plane_n_y, plane_n_x});
    group.accept_mask.fill(false);
    group.accept_mask(0, 0, 0) = true;
    group.selfnoise_offset.resize({plane_n_y, plane_n_x});
    group.selfnoise_offset.fill(0.F);
    group.selfnoise_raw.resize({plane_n_y, plane_n_x});
    group.selfnoise_raw.fill(1.F);
    group.dde_noise.resize({plane_n_y, plane_n_x});
    group.dde_noise.fill(0.F);

    config.subtract_mean = false;
    config.use_rms_noise_estimator = true;
    config.boxsize = 3;
    config.inner_boxsize = 0;
    config.histogram_detail = 1;
    config.max_sigma_bin = 5;
    config.threshold = 0.5F;
    config.beam_size_enabled = false;
    config.band_slices = {BandSlice{.start = 0, .stop = 2, .selfnoise_scale_factor = 1.F, .label = "band"}};
  }
};
} // namespace

TEST_CASE("gpu_process_plane produces the expected plane result", "[gpu][integration]")
{
  const Plane plane;
  const auto &config = plane.config;
  const double expected_extended_box_snr = static_cast<double>(config.max_to_std_ratio_value());
  const auto result = gpu::gpu_process_plane(plane.item, plane.group, config);

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

TEST_CASE("gpu_process_plane skips a band with no nonzero channel", "[gpu][integration]")
{
  // The check is on the host: the band never reaches the GPU, so it adds no median noise, histogram or detection.
  Plane plane;
  plane.item.nonzero_channels = xt::xarray<bool>({false, false});
  const auto result = gpu::gpu_process_plane(plane.item, plane.group, plane.config);

  CHECK(result.detections.size() == 0);
  CHECK(result.median_noises.empty());
  CHECK(result.histograms[0].empty());
  CHECK(result.histograms[1].empty());
  CHECK(result.histograms[2].empty());
  CHECK(result.min_sigma.has_value() == false);
  CHECK(result.max_sigma.has_value() == false);
}

TEST_CASE("gpu_process_plane gives the same result on a reused slot", "[gpu][integration]")
{
  // A slot keeps its buffers from plane to plane, so a plane must not see what the previous one left there. The large
  // plane has three islands and a baseline flux (so a boost of 0.5); the small one has neither, so a stale baseline
  // would show up as a nonzero boost.
  Plane large(6, 8);
  large.group.accept_mask(0, 5, 7) = true;
  large.group.accept_mask(1, 2, 3) = true;
  large.config.baseline_flux = xt::xtensor<float, 2>::from_shape({6, 8});
  large.config.baseline_flux->fill(2.F);
  const Plane small;

  const auto fresh_large = gpu::gpu_process_plane(large.item, large.group, large.config);
  const auto fresh_small = gpu::gpu_process_plane(small.item, small.group, small.config);
  REQUIRE(fresh_large.detections.size() == 3);
  CHECK(fresh_large.detections.boost()[0] == 0.5F);
  REQUIRE(fresh_small.detections.size() == 1);
  CHECK(fresh_small.detections.boost()[0] == 0.F);

  const auto check_same = [](const ProcessPlaneResult &reused, const ProcessPlaneResult &fresh)
  {
    CHECK(reused.detections.peak_y() == fresh.detections.peak_y());
    CHECK(reused.detections.peak_x() == fresh.detections.peak_x());
    CHECK(reused.detections.island_size() == fresh.detections.island_size());
    CHECK(reused.detections.flux() == fresh.detections.flux());
    CHECK(reused.detections.local_noise() == fresh.detections.local_noise());
    CHECK(reused.detections.signal_to_noise() == fresh.detections.signal_to_noise());
    CHECK(reused.detections.signal_to_noise_rms() == fresh.detections.signal_to_noise_rms());
    CHECK(reused.detections.excess_noise_factor() == fresh.detections.excess_noise_factor());
    CHECK(reused.detections.boost() == fresh.detections.boost());
    CHECK(reused.histograms == fresh.histograms);
    CHECK(reused.median_noises == fresh.median_noises);
    CHECK(reused.min_sigma == fresh.min_sigma);
    CHECK(reused.max_sigma == fresh.max_sigma);
  };

  gpu::GpuSlot slot;
  check_same(gpu::gpu_process_plane(slot, large.item, large.group, large.config), fresh_large);
  check_same(gpu::gpu_process_plane(slot, small.item, small.group, small.config), fresh_small);
  check_same(gpu::gpu_process_plane(slot, large.item, large.group, large.config), fresh_large);
}
