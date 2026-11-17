#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "gpu_band.hpp"
#include "gpu_plane.hpp"
#include "process_plane_result.hpp"

namespace plane_tests
{
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

  // The shared inputs are per call: each plane has its own, as each call of run_batch_core_gpu does.
  gpu::GpuSlot slot;
  const auto run = [&slot](const Plane &plane)
  {
    gpu::GpuSharedInputs shared;
    shared.assign(std::span{&plane.group, 1}, plane.config);
    return gpu::gpu_process_plane(slot, shared, plane.item, plane.config);
  };
  check_same(run(large), fresh_large);
  check_same(run(small), fresh_small);
  check_same(run(large), fresh_large);
}
TEST_CASE("GpuSharedInputs empties an input that the next assign does not have", "[gpu]")
{
  Plane with_cutoff;
  xt::xtensor<float, 3> cutoff;
  cutoff.resize({n_channels, n_y, n_x});
  cutoff.fill(1.F);
  with_cutoff.group.snr_cutoff = std::move(cutoff);
  const Plane without_cutoff;

  gpu::GpuSharedInputs shared;
  shared.assign(std::span{&with_cutoff.group, 1}, with_cutoff.config);
  CHECK(shared.groups[0].snr_cutoff.data() != nullptr);
  shared.assign(std::span{&without_cutoff.group, 1}, without_cutoff.config);
  CHECK(shared.groups[0].snr_cutoff.data() == nullptr);
}
} // namespace plane_tests

namespace plane_detections_tests
{
using breifast::ProcessPlaneResult;
using breifast::gpu::append_band_result;
using breifast::gpu::GpuBandHostResult;
using breifast::gpu::GpuBandLabels;
using Catch::Approx;
using Catch::Matchers::Equals;

namespace
{
GpuBandHostResult band_with(const std::size_t count, const float median_noise, const float rms_floor,
                            const float global_mad)
{
  GpuBandHostResult band;
  band.label_count = count;
  band.median_noise = median_noise;
  band.rms_floor = rms_floor;
  band.global_mad = global_mad;
  return band;
}

void resize_detection_vectors(GpuBandHostResult &band)
{
  std::apply([&band](auto &...column) { (column.resize(band.label_count), ...); }, band.islands.tie());
}
} // namespace

TEST_CASE("append_band_result appends detections in the order the bands are appended", "[gpu][plane_detections]")
{
  GpuBandHostResult low = band_with(1, 2.F, 0.5F, 4.F);
  resize_detection_vectors(low);
  low.islands.peak_value[0] = 5.F;
  low.islands.peak_local_noise[0] = 2.F;
  low.islands.peak_y[0] = 1;
  low.islands.peak_x[0] = 2;
  low.islands.island_size[0] = 3;
  low.islands.stamp_extreme[0] = 6.F;
  low.islands.donut_std[0] = 2.5F;
  low.islands.donut_lmad[0] = 1.5F;
  low.islands.beam_chi2[0] = 0.25F;
  low.islands.flux[0] = 10.F;
  low.islands.boost[0] = 1.5F;
  low.islands.noise_at_peak[0] = 3.F;
  low.islands.rms_at_peak[0] = 2.F;
  low.islands.selfnoise_at_peak[0] = 7.F;

  GpuBandHostResult high = band_with(2, 5.F, 1.F, 10.F);
  resize_detection_vectors(high);
  high.islands.peak_value = {8.F, 1.F};
  high.islands.peak_local_noise = {4.F, 1.F};
  high.islands.peak_y = {3, 6};
  high.islands.peak_x = {4, 7};
  high.islands.island_size = {5, 8};
  high.islands.stamp_extreme = {0.F, 2.F};
  high.islands.donut_std = {0.F, 1.F};
  high.islands.donut_lmad = {0.F, 1.F};
  high.islands.beam_chi2 = {0.F, 0.5F};
  high.islands.flux = {20.F, 2.F};
  high.islands.boost = {0.F, 0.25F};
  high.islands.noise_at_peak = {8.F, 1.F};
  high.islands.rms_at_peak = {0.F, 4.F};
  high.islands.selfnoise_at_peak = {9.F, 3.F};

  // CPU order: the last band first.
  ProcessPlaneResult result;
  append_band_result(high, {"I", "high"}, 2.F, true, result);
  append_band_result(low, {"I", "low"}, 2.F, true, result);
  const auto &detections = result.detections;

  REQUIRE(detections.size() == 3);
  CHECK_THAT(detections.band_label(), Equals(std::vector<std::string>{"high", "high", "low"}));
  CHECK_THAT(detections.peak_y(), Equals(std::vector<long>{3, 6, 1}));
  CHECK_THAT(detections.peak_x(), Equals(std::vector<long>{4, 7, 2}));
  CHECK_THAT(detections.island_size(), Equals(std::vector<long>{5, 8, 3}));

  CHECK(detections.flux()[0] == Approx(20.F));
  CHECK(detections.local_noise_median_ratio()[0] == Approx(4.F / 5.F));
  CHECK(detections.signal_to_noise()[0] == Approx(8.F));
  CHECK(detections.signal_to_noise_rms()[0] == Approx(0.F));
  CHECK(detections.signal_to_noise_extended_box()[0] == Approx(0.F));
  CHECK(detections.signal_to_noise_local_mad()[0] == Approx(0.F));
  CHECK(detections.signal_to_noise_global_mad()[0] == Approx(2.F));
  CHECK(detections.self_noise()[0] == Approx(9.F));
  CHECK(detections.excess_noise_factor()[0] == Approx(8.F));

  CHECK(detections.flux()[1] == Approx(2.F));
  CHECK(detections.local_noise_median_ratio()[1] == Approx(1.F / 5.F));
  CHECK(detections.signal_to_noise_rms()[1] == Approx(2.F));
  CHECK(detections.signal_to_noise_extended_box()[1] == Approx(2.F));
  CHECK(detections.signal_to_noise_local_mad()[1] == Approx(2.F));
  CHECK(detections.signal_to_noise_global_mad()[1] == Approx(0.2F));
  CHECK(detections.excess_noise_factor()[1] == Approx(0.25F));
  CHECK(detections.beam_chi2()[1] == Approx(0.5F));
  CHECK(detections.boost()[1] == Approx(0.25F));

  CHECK(detections.local_noise_median_ratio()[2] == Approx(1.F));
  CHECK(detections.signal_to_noise()[2] == Approx(5.F));
  CHECK(detections.signal_to_noise_rms()[2] == Approx(4.F));
  CHECK(detections.signal_to_noise_extended_box()[2] == Approx(10.F / 3.F));
  CHECK(detections.signal_to_noise_local_mad()[2] == Approx(10.F / 1.5F));
  CHECK(detections.signal_to_noise_global_mad()[2] == Approx(2.5F));
  CHECK(detections.self_noise()[2] == Approx(7.F));
  CHECK(detections.excess_noise_factor()[2] == Approx(1.5F));
  CHECK(detections.beam_chi2()[2] == Approx(0.25F));
  CHECK(detections.boost()[2] == Approx(1.5F));
}

TEST_CASE("append_band_result keeps the CPU histogram, median and sigma order", "[gpu][plane_detections]")
{
  // The last band, appended first, passed the filters with sigmas in [0.5, 2]; the first band, appended last, had no
  // pixel pass (infinite minimum) and leaves the range alone, but its histograms are the ones the plane keeps.
  GpuBandHostResult last_band;
  last_band.histogram0 = {7ULL, 8ULL};
  last_band.histogram1 = {9ULL, 10ULL};
  last_band.histogram2 = {11ULL, 12ULL};
  last_band.median_noise = 20.F;
  last_band.min_sigma = 0.5F;
  last_band.max_sigma = 2.F;

  GpuBandHostResult first_band;
  first_band.histogram0 = {1ULL, 2ULL};
  first_band.histogram1 = {3ULL, 4ULL};
  first_band.histogram2 = {5ULL, 6ULL};
  first_band.median_noise = 10.F;
  first_band.min_sigma = std::numeric_limits<float>::infinity();
  first_band.max_sigma = std::numeric_limits<float>::lowest();

  ProcessPlaneResult result;
  append_band_result(last_band, {"I", "last"}, 2.F, true, result);
  append_band_result(first_band, {"I", "first"}, 2.F, true, result);

  CHECK(result.detections.size() == 0);
  CHECK_THAT(result.histograms[0], Equals(std::vector<long>{1, 2}));
  CHECK_THAT(result.histograms[1], Equals(std::vector<long>{3, 4}));
  CHECK_THAT(result.histograms[2], Equals(std::vector<long>{5, 6}));
  CHECK_THAT(result.median_noises, Equals(std::vector<double>{20.0, 10.0}));
  REQUIRE(result.min_sigma.has_value() == true);
  REQUIRE(result.max_sigma.has_value() == true);
  CHECK(*result.min_sigma == 0.5F);
  CHECK(*result.max_sigma == 2.F);
}
} // namespace plane_detections_tests
